# Cardiovascular phenomenological EP foundation

The foundation route is an explicitly **phenomenological** excitation model on a
fixed affine P1 simplex mesh. It is useful for activation-wave studies,
integration tests, and workflows that do not require a biophysical cell model.
The activation and recovery coordinates are dimensionless. They are not membrane
voltage or ionic concentrations, and the source term is not a physical ionic or
transmembrane current density.

## Numerical and unit contract

The kernel uses millimetres and milliseconds in this route:

| Quantity | Kernel unit | Meaning |
| --- | --- | --- |
| mesh coordinates | `mm` | fixed reference coordinates |
| time, step, LAT | `ms` | serial logical time |
| diffusivity tensor K | `mm^2/ms` | phenomenological smoothing coefficient |
| activation, recovery | `1` | dimensionless state coordinates |
| activation source | `ms^-1` | phenomenological activation rate |
| chord CV | `mm/ms` | directed chord distance divided by LAT difference |

`AlievPanfilovParameters(a, b, k, epsilon0, mu1, mu2, tau)` implements exactly

```text
du/dt = (k u (u-a) (1-u) - u r) / tau + s
dr/dt = (epsilon0 + mu1 r / (u+mu2))
        (-r - k u (u-b-1)) / tau
```

where `tau` is in milliseconds and `s` is in inverse milliseconds. The parameter
identity includes all seven coefficients, the denominator tolerance, the unit
contract, and the equation convention. Evaluation records the minimum absolute
value and singular count for `u + mu2`. A singular or nonfinite candidate returns
zero candidate rates and non-success evidence; it is never silently accepted.

## Prepare an affine P1 tetrahedral route

Build the mesh and its scalar `activation` field with the generic finite-element
substrate. The application plan verifies that the field is vertex-associated H1
P1, that coordinate DOFs coincide with simplex vertices, and that the supplied K
has one tensor per cell.

```python
import jax.numpy as jnp
import phydrax as phx

mesh = phx.discretization.CellMesh.from_tetrahedra(coordinates_mm, tetrahedra)
fem = phx.discretization.FiniteElementPlan(
    mesh,
    phx.discretization.FiniteElementFieldSpec(
        "activation",
        phx.discretization.lagrange_element("tetrahedron", 1),
    ),
).prepare()

cell_fibres = jnp.asarray(fibre_vectors)
diffusivity = phx.applications.cardiovascular.electrophysiology.CellwiseDiffusivity.from_fibers(
    cell_fibres,
    longitudinal_mm2_per_ms=0.5,
    transverse_mm2_per_ms=0.05,
)
reaction = phx.applications.cardiovascular.electrophysiology.AlievPanfilovParameters(
    0.05, 0.15, 8.0, 0.002, 0.2, 0.3, 12.9
)
pulse = phx.applications.cardiovascular.electrophysiology.CellStimulusPulse(
    selected_cell_ids,
    0.0,
    1.0,
    3.0,
)
plan = phx.applications.cardiovascular.electrophysiology.PhenomenologicalMonodomainPlan(
    fem,
    diffusivity,
    reaction,
    pulses=(pulse,),
)
runtime = plan.prepare(0.02)
```

The fibre construction uses
`K = d_transverse I + (d_longitudinal-d_transverse) f f^T` after normalizing
`f`. Reversing a fibre leaves K and its identity unchanged. Direct tensor input
must be finite, symmetric, and positive semidefinite. The sign convention is
explicit: positive K smooths activation through `-M_lumped^-1 S u`.

Preparation uses the generic FEM geometry and sparse assembly routes. It computes
all cell gradients, quadrature, the anisotropic stiffness, the row-sum-lumped P1
mass, and pulse projection vectors once. A selected constant cell source is
projected as its exact P1 L2 load, then divided by the nodal lumped mass. No matrix
or cell load is assembled during a time step.

The runtime state is one flat `ProductStateGeometry` containing fixed-size
`activation` and `recovery` blocks plus the immutable prepared `runtime_id`.
Evaluation, splitting, commit, identity, and checkpoint operations reject a
same-shaped state from another runtime. Connectivity and FEM preparation are the
fixed-topology differentiation boundary. Use `runtime.split(state)` when named
blocks are needed.

## Pulse and time-step semantics

Pulse intervals are half-open: `[start_ms, stop_ms)`. Both endpoints must be exact
integer multiples of `dt_ms`; preparation rejects a misaligned endpoint. The
pulse selected for logical step `n` is held constant through all three SSPRK33
stages for interval `[n*dt_ms, (n+1)*dt_ms)`. A pulse ending at `t` is therefore
off for the step beginning at `t`, including every internal stage.

The prepared runtime computes the largest eigenvalue of
`M_lumped^-1/2 S M_lumped^-1/2` with the PhydraX Hermitian spectral substrate.
`dt_ms` must not exceed the SSPRK33 negative-real-axis diffusion-only bound
`2.5127453266183286 / lambda_max`. This bound covers diffusion only; reaction
accuracy and application-specific resolution remain the caller's responsibility.

```python
state = runtime.initialize(
    jnp.zeros(runtime.plan.node_count),
    jnp.zeros(runtime.plan.node_count),
)
candidate = runtime.evaluate(state)
state = runtime.commit(candidate, state)
```

Every stage records finite-state and Aliev--Panfilov singularity evidence. Commit
requires success and exact agreement between the candidate source and the supplied
current state. A failed stage or mismatched candidate preserves the current state.

## Online LAT and directed chord CV

LAT and chord CV are observation operations. Keep them outside the SSPRK33 right
hand side so observation bookkeeping cannot affect solver arithmetic or gradients.

```python
ep = phx.applications.cardiovascular.electrophysiology
lat_plan = ep.ActivationObservationPlan(
    runtime.plan.node_count,
    observed_node_ids,
    threshold=0.5,
)
activation, _ = runtime.split(state)
lat_state = ep.initialize_activation_observation(
    lat_plan, activation, time_ms=float(state.time_ms)
)

candidate = runtime.evaluate(state)
next_state = runtime.commit(candidate, state)
activation, _ = runtime.split(next_state)
lat_candidate = ep.evaluate_activation_observation(
    lat_plan, lat_state, activation, next_state.time_ms
)
lat_state = ep.commit_activation_observation(lat_candidate, lat_state)
result = ep.activation_observation_result(lat_plan, lat_state)
```

Each selected node has an independent first-upward-crossing branch. Crossing time
is linearly interpolated between the two committed samples. Nodes that have not
crossed are censored as `activated=False` with `activation_times_ms=NaN`; ordinary
online censoring is not an observer failure.

A chord plan binds source and target node IDs to the LAT plan and uses a positive
directed distance. CV succeeds only when both LATs exist and the target LAT is
strictly later:

```python
chord = ep.ChordConductionVelocityPlan.from_coordinates(
    lat_plan, coordinates_mm, source_node_id, target_node_id
)
cv = ep.evaluate_chord_conduction_velocity(chord, result)
```

`velocity_mm_per_ms` is numerically equal to metres per second, but remains labeled
with its kernel unit and is a chord estimate rather than local wave speed.

## Checkpoint, restart, and replay identity

The serial checkpoint route uses the generic lifecycle array archive. It writes
separate activation, recovery, time, and step shards, with per-shard digests and
byte counts. The checkpoint manifest binds the monodomain plan, prepared runtime,
SSPRK execution identity, state layout, and optional parent checkpoint.

```python
archive = ep.write_monodomain_checkpoint(runtime, state, "state.phx")
restored = ep.read_monodomain_checkpoint(runtime, "state.phx")
replayed = ep.run_monodomain_steps(runtime, restored, 100)
```

Opening validates the archive container, inventory, checksums, lifecycle manifest,
all runtime identities, payload shapes, finite state, and the invariant
`time_ms == step_index * dt_ms` before returning state. Corruption, an incomplete
archive, or a runtime mismatch fails closed. `monodomain_state_identity` binds
the state's immutable runtime identity, exact array bytes, and logical time to
the prepared `runtime_id`; an uninterrupted serial run and checkpoint/restart
replay have the same identity.

## Qualification and raw benchmark

Run the tetrahedral slab qualification when validating this route:

```console
python tools/cardiovascular_ep_foundation_qualification.py \
  --cubes 4 --dt-ms 0.02 --steps 500 --pulse-ms 1.0
```

It requires propagation across the slab, successful branchwise LAT and chord CV,
positive row-sum masses, symmetric stiffness, satisfaction of the diffusion bound,
and bitwise-identical checkpoint replay.

The raw performance benchmark reports preparation, compilation/warmup, SSPRK33
step throughput, storage, and exact runtime identity:

```console
python benchmarks/cardiovascular_monodomain.py \
  --cubes 32 --dt-ms 0.02 --steps 200
```
