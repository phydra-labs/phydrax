# Atomistic learning

This guide covers the energy-learning workflow for finite molecules and periodic
crystals, cells, and slabs. It uses the existing material-particle, sparse-relation,
and `GraphIR` substrates rather than introducing another entity or graph system.
PaiNN, low-degree Cartesian NequIP, and standard MACE share one structure,
topology, prediction, training, and artifact surface. Forces are always the
negative position gradient of one scalar energy. Stress is available on request
for cases with a periodic cell. None of these models claims long-range
electrostatics or molecular-dynamics stability. Conservative atomistic simulation
is a separate prepared execution path, documented in the
[atomistic dynamics guide](guides_atomistic_dynamics.md). MACE-specific basis,
tabulation, acceleration, and rights contracts are in the
[native MACE execution guide](guides_mace_execution.md).

## Scale and atom identity are part of the input

Every structure carries an `AtomisticScaleContract` built from exact
`UnitDefinition` values. The contract requires length and ordinary energy in one
reference system and fingerprints the energy as that of one simulated system.
Molar energy is deliberately a different dimension and is never silently combined.

```python
import phydrax as phx
from phydrax.units import ANGSTROM, ELECTRONVOLT

scale = phx.atomistic.AtomisticScaleContract(ANGSTROM, ELECTRONVOLT)
water = phx.atomistic.AtomicStructure(
    [8, 1, 1],
    [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
    [15.999, 1.008, 1.008],
    scale,
    particle_ids=[100, 101, 102],
)
```

`AtomicStructure` prepares a `ParticleSetPlan`/`ParticleDiscretization`, so
stable IDs, masses, and the active mask have the same semantics as other
material-particle methods. `AtomicStructure` contains elements only.
`AtomisticBatch` and `AtomisticSystemPlan` additionally carry `element_mask`:
active elements require positive atomic numbers, while active non-element
particles and inactive padding use atomic number zero. `atom_type_ids` remains
an independent interaction-species contract. `AtomisticBatch.from_structures`
pads without changing an active atom's ID or mass. All cases in one batch must
have the same exact scale identity.

Cell and periodic-axis arrays select the geometry. A batch without periodic
axes is a free-space molecule. A batch with periodic axes is evaluated over
explicit lattice images of its `(3, 3)` row cell `H`, including partially
periodic cases. PaiNN, NequIP, and MACE all declare orthorhombic, triclinic,
and cell-derivative capability. A model never reinterprets a periodic structure
as a free-space molecule or the reverse.

## Prepared graph topology and resource contracts

Graph topology and numerical geometry are separate boundaries. An
`AtomisticGraphTopology` holds integer routes, explicit integer image shifts
`n`, candidate membership, stable receiver-major route IDs, and the prepared
streamed schedule for one topology epoch. It is prepared once on the host from
concrete positions and cells. Binding then only computes the directed
displacement `d_e = x[receiver] - x[sender] + n_e @ H[case_e]`, distances,
directions, cutoff masks, and the image certificate. Binding performs no route
discovery, sort, or hash, so it can be differentiated and compiled. Integer
routes, images, and topology epochs are not differentiable. Rebuilding a
topology is an outer data or preparation event.

Image topologies contain every admitted image route exactly once per direction:

- the pair `(i, j, n)` reverses to `(j, i, -n)`;
- nonzero self images `(i, i, n != 0)` are routes; only `(i, i, 0)` is excluded;
- nonperiodic axes carry zero integer components;
- no candidate joins two cases of a batch.

Classical pair-once terms keep their own minimum-image guards. See the
[dynamics guide](guides_atomistic_dynamics.md) for runtime image neighborhoods,
Verlet reuse, wrap handling, and certificates.

`AtomisticGraphExecutionPlan` declares the resources explicitly:

- `backend="dense"` is the bounded named reference. Finite batches use the
  historical directed all-pairs layout, and periodic batches use all pairs
  times a complete image stencil. The plan requires `maximum_dense_atoms`, and
  a batch above it is rejected.
- `backend="particle"` searches image routes with the case-batched fractional
  cell list. It forbids `maximum_dense_atoms`.
- Periodic batches and the particle backend require `image_capacity`, a
  `ParticleImageCapacity` whose cell occupancy, stored directed edges, receiver
  degree, and integer image stencil are separately charged budgets.
  `maximum_candidate_slots` guards the derived candidate work.
- `streamed` is the `StreamedRelationPlan` (receiver tile, edge fragment,
  channel cap, accumulation, replay). Its schedule is prepared once per
  topology. Omitting it uses the default plan.
- `maximum_neighbors` is a separate runtime degree capacity.

No capacity is implemented by truncation, clipping, repair, or partial neighbor
selection. `AtomisticGraph.require_success` and direct energy evaluation fail
closed. `energy_and_forces` instead returns `valid=False`, status
`NEIGHBOR_OVERFLOW`, and NaN values, so a batch pipeline retains typed failure
evidence.

```python
execution = phx.atomistic.AtomisticGraphExecutionPlan(
    16,
    maximum_dense_atoms=32,
)
batch = phx.atomistic.AtomisticBatch.from_structure(water)
graph = phx.atomistic.realize_atomistic_graph(
    batch,
    execution,
    cutoff=5.0,
)
```

`realize_atomistic_graph` without a topology is the finite dense realization.
Periodic batches pass a topology from `prepare_atomistic_graph_topology`:

```python
crystal_execution = phx.atomistic.AtomisticGraphExecutionPlan(
    64,
    backend="particle",
    image_capacity=phx.discretization.ParticleImageCapacity(
        maximum_particles_per_cell=16,
        maximum_edges=4096,
        maximum_degree=128,
        maximum_images=343,
    ),
)
periodic_water = phx.atomistic.AtomicStructure(
    [8, 1, 1],
    [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
    [15.999, 1.008, 1.008],
    scale,
    cell=[[4.2, 0.0, 0.0], [0.9, 4.0, 0.0], [0.5, -0.6, 4.4]],
    periodic_axes=[True, True, True],
)
crystal = phx.atomistic.AtomisticBatch.from_structure(periodic_water)
topology = phx.atomistic.prepare_atomistic_graph_topology(
    crystal, crystal_execution, cutoff=5.0, skin=0.5
)
crystal_graph = phx.atomistic.bind_atomistic_graph(
    topology,
    crystal_execution,
    crystal.positions.reshape((-1, 3)),
    cutoff=5.0,
    cell_vectors=crystal.cells,
)
```

The cell above is smaller than twice the cutoff. Its topology therefore holds
repeated and self-image routes that a minimum-image graph would miss.
`prepare_atomistic_graph_topology` refuses an image stencil, edge count, or
degree beyond the declared capacity before allocation.

## PaiNN scalar/vector interactions

`PaiNNPotential` embeds the declared `AtomisticSpeciesKind` into invariant
scalar features. Atomic models use atomic numbers; molecular coarse models use
explicit atom-type IDs. Each interaction combines a smooth sinusoidal radial
basis and cosine cutoff with scalar messages and Cartesian vector messages.
Channel maps are native PhydraX `Linear` layers using the parameter-transform
contract; contractions use `opt_einsum.contract`. Vector channels only undergo
inner products, scalar gating, and multiplication by relative unit directions.
The total energy is a masked sum of invariant per-atom scalar readouts and is
therefore translation-, rotation-, and atom-permutation-invariant.

Messages and the residual atomwise update run through the shared streamed
relation: each directed edge produces one filtered message, and each receiver
applies its update once after summing all of its messages. Periodic batches use
the same formulas over explicit image routes. Cell vectors enter only through
the image displacements.

```python
import jax.random as jr

potential = phx.nn.atomistic.PaiNNPotential(
    scale,
    cutoff=5.0,
    feature_count=64,
    interaction_count=3,
    radial_basis_count=20,
    key=jr.key(0),
)
prediction = phx.atomistic.energy_and_forces(potential, batch, execution)
```

## Low-degree Cartesian NequIP interactions

`NequIPPotential` is a drop-in alternative under the same `AtomicStructure`,
`AtomisticBatch`, `energy_and_forces`, and `fit_atomistic_potential` contracts.
It embeds species into invariant scalar channels, forms scalar, vector, and
symmetric-traceless rank-two edge features, and applies weighted Cartesian O(3)
tensor products. Receiver aggregation, species-conditioned equivariant self
connections, and parity-safe gates update the node state. Only invariant scalar
channels enter the masked per-atom energy readout.

```python
nequip = phx.nn.atomistic.NequIPPotential(
    scale,
    cutoff=5.0,
    feature_count=32,
    interaction_count=3,
    radial_basis_count=20,
    key=jr.key(2),
)
nequip_prediction = phx.atomistic.energy_and_forces(nequip, batch, execution)
```

NequIP keeps the physical Cartesian `O3Representation`: scalar/pseudoscalar,
vector/pseudovector, and symmetric-traceless tensor/pseudotensor blocks of degree
zero through two. Its block order, basis, and learned parameter shapes are
unchanged. Its `O3TensorProductPlan` resolves every legal degree/parity
instruction, the canonical fully connected `uvw` multiplicity weights,
component normalization, parameter count, coefficient storage, scalar
contraction work, resource limits, and a content ID before `O3TensorProduct`
prepares coefficients or weights.

The same tensor-product owner also accepts the general real-irrep
`O3IrrepLayout` and `uvu` incidence used by MACE. The two layouts are distinct
contracts, not aliases. NequIP itself remains a degree-at-most-two Cartesian
model; arbitrary degrees belong to the general layout. Radial networks emit one
coefficient for every actual tensor-product instruction weight rather than one
scalar per output block. Each directed edge's radially weighted tensor-product
message streams through the shared relation, and the self-connection plus gate
runs once per receiver. Padded nodes and edges are masked at embedding,
edge-feature, message, aggregation, and readout boundaries.

NequIP is independently derived in this Cartesian convention. It is not
e3nn-compatible NequIP and makes no claim for higher degrees, long-range
electrostatics, or molecular-dynamics stability.

## Standard MACE

`MACEPotential` is the native trainable standard MACE family:

- real-agnostic plain, residual, density-normalized, and density-residual
  interactions;
- symmetric products that keep their original weights `W` over fixed coupling
  bases `U`;
- linear and nonlinear readouts, multihead scale-shift energies, and atomic
  reference energies;
- optional ZBL and Agnesi transforms.

It uses the same prepared topologies, prediction, training, dynamics, and
artifact boundaries as PaiNN and NequIP.

```python
mace = phx.nn.atomistic.MACEPotential(
    scale,
    phx.nn.atomistic.MACEArchitecture(
        species=(1, 8),
        cutoff=3.0,
        radial_basis_count=4,
        cutoff_power=5,
        channel_count=4,
        hidden_degree=1,
        edge_degree=2,
        interactions=("real-agnostic", "real-agnostic-residual"),
        correlations=(2, 2),
        radial_widths=(8,),
        readout_width=4,
        average_neighbor_count=2.0,
    ),
    atomic_energies=jnp.asarray([[-1.0, -2.0]], dtype=jnp.float64),
    key=jr.key(3),
)
mace_prediction = phx.atomistic.energy_and_forces(
    mace, crystal, crystal_execution, compute_stress=True
)
```

The [native MACE execution guide](guides_mace_execution.md) covers the admitted
architecture, basis and parameter fidelity, prepared exact and tabulated
inference, the accelerated coupling, and the rights boundary. All MACE routes
are unreleased candidates.

## Forces, stress, and provenance

The force is the negative position gradient of one scalar total-energy closure.
Integer candidates, images, and the topology identity stay frozen during the
derivative, and the smooth cutoff controls interaction support. There is no
direct-force output.

`energy_and_forces(..., compute_stress=True)` also differentiates that energy
with respect to a homogeneous strain. Row lattice vectors deform as `H @ F.T`
with `F = I + strain`, at fixed fractional coordinates and fixed integer
images. The result is the tensile-positive stress `sym(dE/d strain) / |det H|`
in energy per cubic length; pressure is `-trace(stress) / 3`. A self-image
route can contribute stress while its positional forces cancel.
`AtomisticStressConvention` names the meaning of each case:

- `CAUCHY_TENSION_POSITIVE` for fully periodic cells;
- `CAUCHY_TENSION_POSITIVE_EMBEDDING_VOLUME` for partially periodic cases,
  where the full invertible 3×3 cell is the caller's declared embedding and
  the value is cell-volume-normalized, not an intrinsic surface or line
  stress.

A stress request needs explicit periodic cell data and a potential with
`cell_derivative` capability; otherwise it is refused. `stress_case_mask`
selects cases of a mixed finite/periodic batch. Unrequested cases carry NaN
stress with `stress_available=False`, and a finite case cannot be requested.

`AtomisticPrediction` contains energy, per-atom energy, forces, requested
stress and its per-case availability, validity/status, overflow evidence,
maximum neighbor work, net force, center-of-mass net torque, scale, and
provenance. Net force and torque diagnose numerical equivariance defects. They
are not silently projected to zero, and for periodic cases they are diagnostics
of the unwrapped arrays only. A failed case returns NaN values and NaN
derivatives, never a successful zero.

`atomistic_energy_derivatives(potential, batch, execution, positions,
topology=..., cell_vectors=..., compute_forces=..., compute_stress=...)` is the
numerical boundary underneath. It performs no host identity work, topology
discovery, or hashing, so it can be called under `jit`, JVP/VJP, and mixed
force-loss or stress-loss parameter differentiation. `energy_and_forces` is
its host-facing wrapper: it prepares a topology at the potential's cutoff when
none is supplied and records provenance.

`AtomisticPrecisionPolicy` separately declares coordinate, interaction,
reduction, and output dtypes. `AtomisticProvenance` records:

- the scale, precision, and architecture identities;
- the graph execution plan and graph topology identities;
- whether the case is periodic and its stress conventions;
- the canonical `phydrax.NumericRevision` of the exact evaluated parameters.

`phx.atomistic.atomistic_potential_revision(potential)` derives that revision on
the host from the potential's PARAMETER role lane: its semantic provenance names
the architecture and force method, and its numeric content is every parameter
leaf keyed by tree path. Fixed and model-state leaves, such as MACE coupling
bases, never enter it. Because the identity is computed from the current
parameters, an external Equinox or Optax update is reflected immediately and no
checkpoint step exists:

```text
updated = eqx.tree_at(lambda potential: potential.embedding, potential, embedding)
revision = phx.atomistic.atomistic_potential_revision(updated)
prediction = phx.atomistic.energy_and_forces(updated, batch, execution)
```

The revision is a host boundary. `energy_and_forces` records it, so call it with
concrete parameters rather than on a potential traced by `jit`, `vmap`, or
`grad`. Differentiate through `atomistic_energy_derivatives` or the training
entry points.

## Energy, force, and stress training

Training is domain-specific; it does not add a generic trainer. Every
`AbstractAtomisticPotential` uses the same `AtomisticTrainingProblem`, which
holds one training split and one optional validation split
(`AtomisticSupervisionSplit`, role `"training"` or `"validation"`).

Each split binds a native batch to a frozen candidate graph topology:

- Without a supplied topology, the problem prepares one on the host from
  `cutoff` and `skin`. `cutoff` is required in that case.
- `training_topology`/`validation_topology` accept a caller-prepared topology,
  for example a spatial periodic-image topology. Its search radius must equal
  `cutoff + skin` when a cutoff is also given, and must cover the potential's
  cutoff.
- A topology is never rebuilt during an optimizer step.

Labels are optional per kind:

- energies in system energy units;
- forces in energy per length;
- stress as the tensile-positive `(1/V) dE/d strain` tensor of the supplied cell.

Each kind has its own mask. Stress applies only to cases with a periodic axis
and a finite nonsingular cell, so a stress mask can select the periodic cases
of a mixed finite/periodic batch. Validation must supervise the same target
kinds as training. Stress supervision requires a potential with cell-derivative
capability.

The losses are normalized mean squares:

- energy per atom;
- each selected Cartesian force component;
- each selected stress component.

`energy_weight`, `force_weight`, and `stress_weight` combine them. Each scale
can be supplied (`energy_scale`, `force_scale`, `stress_scale`) or fitted from
the training split only: standard deviation of per-atom energy and RMS of the
selected force and stress components, floored by `normalization_floor`. A
validation split never contributes to fitted values. The parameter gradient of
a force or stress loss is a mixed coordinate/parameter or strain/parameter
derivative of the same scalar energy; no Jacobian or Hessian is materialized.

```text
problem = phx.atomistic.AtomisticTrainingProblem(
    training_batch,
    execution,
    cutoff=5.0,
    training_energy=training_energy,
    training_forces=training_forces,
    training_stress=training_stress,
    training_stress_mask=training_stress_mask,
    validation_batch=validation_batch,
    validation_energy=validation_energy,
    validation_forces=validation_forces,
    validation_stress=validation_stress,
    validation_stress_mask=validation_stress_mask,
)
policy = phx.atomistic.AtomisticTrainingPolicy(
    maximum_steps=1_000,
    learning_rate=1e-3,
    energy_weight=1.0,
    force_weight=100.0,
    stress_weight=10.0,
    validation_every=10,
    patience=20,
)
events = []
session = phx.execution.IterationSession(
    "atomistic-fit",
    sinks=(
        phx.execution.CallableIterationSink(
            lambda event: events.append(event),
            "fit-events",
        ),
    ),
)
result = phx.atomistic.fit_atomistic_potential(
    potential,
    problem,
    policy,
    key=jr.key(1),
    session=session,
)
```

`examples/atomistic_mace.py` is a complete runnable version. It labels strained
periodic cells with a teacher MACE through `energy_and_forces(...,
compute_stress=True)` and fits a differently initialized student on
energy/force/stress.

Every update is one full-batch Adam attempt of the shared accepted-update
training kernel, with `MODEL` root authority and one data-fit objective. The
fit loop uses the shared `TrainingController` for the master key, selection,
patience, and progress. Typed lifecycle events are delivered through an
explicit `IterationSession`; observation sinks cannot alter training, while a
separate `IterationHostControl` may request stopping at an update boundary.

`AtomisticTrainingResult` retains:

- final and best potentials;
- the committed training-kernel state (parameters, Adam state, root key, and
  cursors);
- the fitted `AtomisticTrainingNormalization`;
- the training, energy, force, and stress loss histories;
- validation values and steps, progress, status, and termination identity;
- the problem, policy, continuation, capability, and checkpoint identities.

Continue deterministically by raising the total step ceiling and passing
`continuation=result`. A continued host-controlled run must restore the
persisted session identity and cursor.

Unsupported periodic, cell, or target configurations are refused before any
optimizer state exists. A nonfinite training loss or gradient, including
capacity overflow of a frozen topology, rolls the attempt back and terminates
with `AtomisticStatus.NONFINITE`. An update whose post-update training loss is
nonfinite is discarded the same way, so `potential` is always the last finite
accepted state. A model selected by patience or host iteration control retains
`STOPPED_EARLY`. Neither case is reported as an ordinary maximum-step
completion.

The supervision split, problem, policy, and result identities now cover
topology, stress, and the new weights and scales. Identities computed by
earlier versions are not reproduced, and no mapping from them is provided.
Likewise, `AtomisticGraphExecutionPlan.plan_id` now covers the backend, image
capacity, streamed plan, and candidate-slot guard.

## Native model artifacts and restarts

Native model persistence is pickle-free and needs no optional provider package.
`write_atomistic_model_artifact(path, model, source=..., licenses=...)`
validates the model with its registered domain validator and atomically writes
a canonical array archive:

- the registered structure recipe, with no Python module paths;
- the dynamic leaves;
- verified identities: `architecture_id`, `method_id`, `structure_id`,
  `semantic_id`, the parameter `numeric_revision`, a `content_id` over every
  leaf including fixed coefficient, basis, and table leaves, conversion
  provenance, license identifiers, and runtime versions.

`read_atomistic_model_artifact(path, numeric_revision_id=...)` restores the
model in bounded, fail-closed order:

1. exact member, byte, shape, and dtype inventory before allocation
   (`ATOMISTIC_MODEL_ARTIFACT_LIMITS`);
2. a registered type and field guard;
3. one typing validation of the complete restored value;
4. the model class's own scientific validators;
5. recomputation of every recorded identity.

Serialized identities and success flags are never trusted as evidence of
themselves. A structurally readable artifact that fails a scientific or
identity check raises `AtomisticModelArtifactError`. Licenses are recorded, and
redistribution permission is never inferred. Exact `MACEPotential` and prepared
`PreparedMACEPotential` forms are registered.
`register_atomistic_model_artifact` admits another model class only together
with its owning validator, revision, and exact-potential functions.

```text
manifest = phx.atomistic.write_atomistic_model_artifact(
    "water-mace.phydrax", mace, licenses=("MIT",)
)
artifact = phx.atomistic.read_atomistic_model_artifact(
    "water-mace.phydrax",
    numeric_revision_id=manifest.numeric_revision.revision_id,
)
restored_prediction = phx.atomistic.energy_and_forces(
    artifact.model, crystal, crystal_execution
)
```

Continuation across processes uses explicit bundles:

- `write_atomistic_training_restart(directory, result, policy)` atomically
  publishes the artifact of `result.potential`, the training kernel checkpoint
  (optimizer state, root key, cursors, best potential, and histories), and a
  `restart.json` digest receipt as one bundle. A failed publication preserves
  the previous resumable bundle. The training data recipe is not stored.
  `read_atomistic_training_restart(directory, problem, policy)` rebuilds against
  the same `AtomisticTrainingProblem` and refuses mismatched receipts,
  artifacts, kernel states, and obsolete receipt-free directory layouts.
  `write_atomistic_training_checkpoint`/`read_atomistic_training_checkpoint`
  are the kernel-only boundary. Normalization is refitted on restore and must
  reproduce its identity.
- `write_atomistic_restart(path, plan, state, model=...)` bundles the model
  artifact with dynamics state. A fresh process restores the model with
  `read_atomistic_restart_model`, rebuilds its dynamics, and resumes with
  `read_atomistic_restart`. Any altered model, source binding, system,
  integrator, thermodynamic table, scope, or graph preparation refuses.
  Compiled caches are not part of restart correctness.
  `examples/periodic_mace_dynamics.py` resumes from such a bundle and compares
  the result with the uninterrupted trajectory.

## Local rMD17 archives

`load_rmd17_npz` only reads a user-provided local archive. It accepts common
rMD17 field names for nuclear charge, coordinates, energy, force, and optional
original sample indices. The boundary declares the source as angstrom and
kilocalorie-per-mole, then converts coordinates to the target length and converts
molar energies and forces to ordinary single-system energy with the exact Avogadro
constant from the recorded `codata-2018` constant set. The default target scale is
angstrom/electronvolt. The source units and Avogadro provenance are retained in
the dataset identity, together with dalton mass provenance; molar and ordinary
energy never become generally convertible. The parser performs no download and
does not import a foreign atomistic framework.

`split_rmd17` makes deterministic, disjoint train/validation/test indices and
fingerprints the exact dataset, seed, and index arrays. The default sizes are
950/50/1000. `RMD17Dataset.take` returns a native `AtomisticBatch` and aligned
energy/force arrays.

The developer benchmark tool `tools/atomistic_rmd17_benchmarks.py` accepts a
local NPZ or an explicit URL plus mandatory SHA-256. For every seed it uses the
same split, optimizer/loss policy, cutoff, capacity, feature width, interaction
count, and radial basis for PaiNN and NequIP. It records errors, equivariance
defects, compile/steady timings, host memory, model parameters, dense-candidate
and active-neighborhood work, predeclared gates, per-model and paired summaries,
tensor-product plan evidence, and environment provenance. It prints an artifact
only when run; no data or benchmark result is bundled with PhydraX.
