# Atomistic learning

This guide covers the energy-learning workflow for finite molecules and periodic
crystals, cells, and slabs. It uses the existing material-particle, sparse-relation,
and `GraphIR` substrates rather than introducing another entity or graph system.
PaiNN, low-degree Cartesian NequIP, and standard MACE share one structure, graph
layout, prediction, training, and artifact surface. Forces are always the
negative position gradient of one scalar energy. Stress is available on request
for cases with a periodic cell. None of these models claims long-range
electrostatics or molecular-dynamics stability. Conservative atomistic simulation
is a separate prepared execution path, documented in the
[atomistic dynamics guide](guides_atomistic_dynamics.md). MACE-specific basis,
tabulation, acceleration, and rights contracts are in the
[native MACE execution guide](guides_mace_execution.md), and the MACELES,
AIMNet2, and UMA families with their conditions and source gates are in the
[model-family guide](guides_atomistic_models.md). Streamed structure sources,
loaders, and objectives are in the
[training-data guide](guides_atomistic_training_data.md); native teachers and
on-policy distillation are in the [distillation guide](guides_atomistic_distillation.md).

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

## Graph layouts, payloads, and resource contracts

Graph preparation and numerical geometry are separate boundaries:

- `AtomisticGraphLayout` owns every static executable choice: scale, execution
  plan, case and atom capacities, coordinate dtype, cutoff, skin, the periodic
  axes cases may declare, whether payloads carry cells, and the route-search
  envelope. Its identities are configuration fingerprints and contain no
  material content.
- `layout.prepare(batch)` is the host boundary. From concrete positions and
  cells it prepares an `AtomisticGraphPayload`: an `AtomisticGraphTopology` with
  integer routes, explicit integer image shifts `n`, candidate membership,
  stable receiver-major route IDs, and the streamed schedule of one topology
  epoch, together with the payload's positions, element mask, cells, and
  periodic axes. Material case words, stable particle IDs, species, masses, and
  masks are dynamic topology leaves, so payloads of one layout share one static
  tree structure and one executable.
- `payload.graph(cutoff=..., positions=None, cell_vectors=None,
  image_counts=None, selection=None)` binds geometry. It only computes the
  directed displacement `d_e = x[receiver] - x[sender] + n_e @ H[case_e]`,
  distances, directions, cutoff masks, and the image certificate. Binding
  performs no route discovery, sort, or hash, so it can be differentiated and
  compiled. Integer routes, images, and topology epochs are not
  differentiable; rebuilding a topology is an outer data or preparation event.

`AtomisticBatch` remains the host content and provenance envelope and never
enters a compiled step. `topology.content_id()` is the host-only content
identity of the material and routes.

Image topologies contain every admitted image route exactly once per direction:

- the pair `(i, j, n)` reverses to `(j, i, -n)`;
- nonzero self images `(i, i, n != 0)` are routes; only `(i, i, 0)` is excluded;
- nonperiodic axes carry zero integer components;
- no candidate joins two cases of a batch.

Classical pair-once terms keep their own minimum-image guards. See the
[dynamics guide](guides_atomistic_dynamics.md) for runtime neighborhood
bundles, candidate refresh, wrap handling, and certificates.

`AtomisticGraphExecutionPlan` declares the resources explicitly:

- `backend="dense"` is the bounded named reference. Finite batches use the
  directed all-pairs layout, and periodic batches use all pairs times a complete
  image stencil. The plan requires `maximum_dense_atoms`, and a layout above it
  is rejected.
- `backend="particle"` searches image routes with the case-batched fractional
  cell list. It forbids `maximum_dense_atoms`.
- Periodic layouts and the particle backend require `image_capacity`, a
  `ParticleImageCapacity` whose cell occupancy, stored directed edges, receiver
  degree, and integer image stencil are separately charged budgets.
  `maximum_candidate_slots` guards the derived candidate work.
- `search_cell` (a finite full-rank `(3, 3)` row lattice in scale length
  units) and `search_margin` declare the route-search envelope of every
  reusable layout: they fix the cell-list grid or dense image stencil, and the
  margin widens the prepared stencil. The envelope is a search resource, not an
  extra physical cutoff.
- `streamed` is the `StreamedRelationPlan` (receiver tile, edge fragment,
  channel cap, accumulation, replay). Its schedule is prepared once per
  topology. Omitting it uses the default plan. An ordered accumulation
  (`"deterministic"` or `"compensated"`) also prepares validated geometry
  relations with each topology, so position and cell derivatives, per-case
  energy totals, receiver epilogues, and MACE species embeddings accumulate in
  a fixed order. Bitwise reproducibility on accelerators requires it; the fast
  default keeps vectorized gathers and unordered GPU additions.
- `maximum_neighbors` is a separate runtime degree capacity.

The physical cutoff is never changed by the search. The bound graph activates an
edge only for `distance < cutoff`. The image candidate radius is `cutoff + skin`
rounded outward by one representable value of the layout coordinate dtype
(`layout.search_radius`), so the open candidate ball covers the closed selector
`distance <= cutoff + skin` at that precision instead of losing a pair exactly
on the bound. A bound cutoff above the topology's candidate radius is refused.

No capacity is implemented by truncation, clipping, repair, or partial neighbor
selection. Exhausted occupancy, image, edge, degree, or tile budgets are
per-case `topology.overflow` evidence. A stencil beyond the declared envelope,
a domain violation, a nonfinite lattice, or a foreign material is a scientific
failure recorded in `topology.search_failure`, so growing a capacity can never
admit it. `AtomisticGraph.valid` excludes both. `AtomisticGraph.require_success`
and direct energy evaluation fail closed. `energy_and_forces` instead returns
`valid=False`, status `NEIGHBOR_OVERFLOW` for capacity failures, and NaN values,
so a batch pipeline retains typed failure evidence.

```python
import jax.numpy as jnp

execution = phx.atomistic.AtomisticGraphExecutionPlan(
    16,
    maximum_dense_atoms=32,
)
batch = phx.atomistic.AtomisticBatch.from_structure(water)
payload = phx.atomistic.AtomisticGraphLayout.for_batch(
    batch, execution, cutoff=5.0
).prepare(batch)
graph = payload.graph(cutoff=5.0)
```

`AtomisticGraphLayout.for_batch(batch, execution, *, cutoff, skin=0.0)` derives
a one-batch layout from that batch's capacities, dtype, cells, and periodic
axes. Its envelope is the plan's `search_cell` when declared and otherwise that
batch's own binning lattices. A periodic batch uses the same calls:

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
crystal_payload = phx.atomistic.AtomisticGraphLayout.for_batch(
    crystal, crystal_execution, cutoff=5.0, skin=0.5
).prepare(crystal)
crystal_graph = crystal_payload.graph(cutoff=5.0)
```

The cell above is smaller than twice the cutoff. Its topology therefore holds
repeated and self-image routes that a minimum-image graph would miss.

A layout reused across many batches is constructed directly. Its envelope comes
from `AtomisticGraphLayout(..., search_cell=...)` (one `(3, 3)` lattice or one
per case) or from the plan's `search_cell`; an image layout without one is
refused:

```text
layout = phx.atomistic.AtomisticGraphLayout(
    scale,
    reusable_execution,  # declares search_cell=... and search_margin=...
    case_capacity=8,
    atom_capacity=64,
    coordinate_dtype=jnp.float64,
    cutoff=5.0,
    skin=0.5,
    periodic_axes=(True, True, True),
    cells=True,
)
payload = layout.prepare(batch)
moved = payload.admit(moved_batch)
```

`layout.prepare` refuses a scale, capacity, dtype, cell-class, or undeclared
periodic-axis mismatch on the host. `payload.admit(batch)` binds the payload's
routes to a batch of the same material content at its current geometry.
Particle IDs, species, masses, masks, case words, and periodicity are compared
numerically per case. Moved positions and deformed cells are admitted subject to
the stored image certificate; a foreign-material case fails through
`search_failure`, not `overflow`.

## Typed graph inputs, conditions, and requested outputs

The canonical model boundary is
`model.evaluate_graph(AtomisticGraphInputs(graph, scale, species_kind=...,
conditioning=None, key=None), request)`; both arguments are positional. The
bound graph owns both current coordinates and edges, and species come from its
node data. Species meaning and exact physical scales are declared rather than
inferred from array shape. `AtomisticModelOutputs` returns case energies,
per-atom energies, numerical success, and typed additional observables, each an
`AtomisticObservableValue` with its exact `ValuePort`, level, unit, availability,
and extent. `atom_energy_complete=False` identifies a scoped local
decomposition; it must not be interpreted as the complete total. A
property-only model returns absent energies rather than zeros.

`AtomisticObservableRequest(observables, derivatives=())` names the requested
outputs. A model refuses an unsupported observable through
`request.require_supported(supported, owner)` instead of filling it with a
plausible zero. Derivative requests are typed `phydrax.DifferentiationRequest`
records and are admitted by the model's derivative contract, not by the
observable request.

`AtomisticConditioningSpec` binds named `AtomisticConditionField`s to existing
`ValuePort` identities, case or atom levels, numerical kinds, dtypes, units, and
admissible domains:

```text
from phydrax.units import ELEMENTARY_CHARGE

charge = phx.atomistic.AtomisticConditionField(
    "total_charge",
    phx.atomistic.TOTAL_CHARGE_PORT,
    unit=ELEMENTARY_CHARGE,
    level=phx.atomistic.AtomisticConditionLevel.CASE,
    kind=phx.atomistic.AtomisticConditionKind.CATEGORICAL,
    dtype=jnp.int64,
    categories=tuple(range(-2, 3)),
)
conditioning = phx.atomistic.AtomisticConditioning(
    phx.atomistic.AtomisticConditioningSpec((charge,)),
    {"total_charge": [0]},
    case_count=batch.case_count,
    atom_capacity=batch.atom_capacity,
)
```

Atom conditions use flat case-major rows. Required values have no implicit
defaults: an explicit `defaults=` mapping records which cases used them, and an
absent optional value stays absent. Continuous conditions remain numerical
leaves for differentiation. Categorical codes retain their declared meaning and
are not clamped into the vocabulary. `case_status(atom_mask)` reports nonfinite
values, domain violations, and invalid categories independently, ignoring
inactive atom rows; a violating case is unsuccessful while its neighbors are
unaffected.

Pass `conditioning=` to `AtomisticGraphInputs`, `potential.energy`,
`atomistic_energy_derivatives`, or `energy_and_forces`. Numerical evaluation does
not hash these values; the host-facing prediction records their actual payload
in its provenance. Dimensioned condition and property ports require an explicit
compatible `UnitDefinition`. A source spin-category index and physical spin
multiplicity are distinct ports, even when both have scalar storage. Which
families accept which conditions is listed in the
[model-family guide](guides_atomistic_models.md).

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

The same bound graph can be evaluated directly with an explicit request:

```python
outputs = potential.evaluate_graph(
    phx.atomistic.AtomisticGraphInputs(
        graph, scale, species_kind=phx.atomistic.AtomisticSpeciesKind.ATOMIC_NUMBER
    ),
    phx.atomistic.AtomisticObservableRequest(
        (
            phx.atomistic.AtomisticObservable.ENERGY,
            phx.atomistic.AtomisticObservable.ATOM_ENERGY,
        )
    ),
)
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

`atomistic_energy_derivatives(potential, payload, positions, *,
cell_vectors=None, compute_forces=True, compute_stress=False,
stress_case_mask=None, conditioning=None, key=None, image_counts=None)` is the
numerical boundary underneath. It binds the prepared payload's fixed routes,
performs no host identity work, route discovery, or hashing, and can therefore
be called under `jit`, JVP/VJP, and mixed force-loss or stress-loss parameter
differentiation. Positions are `(case, atom, 3)` in the model coordinate dtype,
and cells default to the payload's cells:

```python
derivatives = phx.atomistic.atomistic_energy_derivatives(
    potential, payload, batch.positions
)
```

`energy_and_forces(potential, structure, execution, *, compute_stress=False,
payload=None, ...)` is its host-facing wrapper. Without a payload it prepares
`AtomisticGraphLayout.for_batch(...)` at the potential's cutoff; a supplied
payload is admitted with `payload.admit(batch)`. It records provenance and
requires concrete model parameters.

`AtomisticPrecisionPolicy` separately declares coordinate, interaction,
reduction, and output dtypes. `AtomisticProvenance` records:

- the scale, precision, method, and architecture identities;
- the graph execution plan, the executable `graph_layout_id`, and the host
  content identity of the graph topology;
- the conditioning, random-realization, and image-count identities of the call;
- whether the case is periodic and its stress conventions;
- the canonical `phydrax.NumericRevision` of the exact evaluated parameters.

`phx.atomistic.atomistic_model_revision(potential)` derives that revision on
the host from the potential's PARAMETER role lane: its semantic provenance names
the architecture and force method, and its numeric content is every parameter
leaf keyed by tree path. Fixed and model-state leaves, such as MACE coupling
bases, never enter it. Because the identity is computed from the current
parameters, an external Equinox or Optax update is reflected immediately and no
checkpoint step exists:

```text
updated = eqx.tree_at(lambda potential: potential.embedding, potential, embedding)
revision = phx.atomistic.atomistic_model_revision(updated)
prediction = phx.atomistic.energy_and_forces(updated, batch, execution)
```

The revision is a host boundary. `energy_and_forces` records it, so call it with
concrete parameters rather than on a potential traced by `jit`, `vmap`, or
`grad`. Differentiate through `atomistic_energy_derivatives` or the training
entry points.

## Energy, force, and stress training

Training is domain-specific; it does not add a generic trainer. Every
conservative native model trains through `fit_atomistic_potential` on an
`AtomisticTrainingProblem` that binds an immutable training loader, an optional
lineage-disjoint validation loader, the graph execution plan, and one declared
objective per target. Sources, mixtures, buckets, cursors, objectives,
schedules, reference energies, and precision are described in the
[training-data guide](guides_atomistic_training_data.md).

```text
source = phx.atomistic.atomistic_training_source(
    training_batch,
    mass_unit=units.mass_unit,
    energy=training_energy,
    forces=training_forces,
    stress=training_stress,
    stress_available=periodic_cases,
)
problem = phx.atomistic.AtomisticTrainingProblem(
    phx.atomistic.atomistic_training_loader(source, batch_size=16),
    execution,
    (
        phx.atomistic.AtomisticEnergyObjective(),
        phx.atomistic.AtomisticForceObjective(
            schedule=phx.atomistic.AtomisticObjectiveSchedule.constant(100.0)
        ),
        phx.atomistic.AtomisticStressObjective(
            schedule=phx.atomistic.AtomisticObjectiveSchedule.constant(10.0)
        ),
    ),
    scale=units.scale,
    cutoff=5.0,
    fields=phx.atomistic.AtomisticTrainingFields(cell="cell", periodic="periodic"),
    validation=phx.atomistic.atomistic_training_loader(
        validation_source, stream="validation"
    ),
)
policy = phx.atomistic.AtomisticTrainingPolicy(
    maximum_steps=1_000, learning_rate=1e-3, validation_every=10, patience=20
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
    potential, problem, policy, key=jr.key(1), session=session
)
```

The default objectives are the structure-balanced per-atom energy mean square,
the per-component force mean square, and the per-component stress mean square.
Scales are declared or fitted from training labels only; validation labels never
contribute, and a fit without available or finite labels, or below the
normalization floor, is refused. Stress applies only to cases with a periodic
axis and a finite nonsingular cell; an available stress label on another case is
refused when its payload is prepared, so declare such cases unavailable. Stress
objectives require a model with cell-derivative capability. The parameter
gradient of a force or stress loss is a mixed coordinate/parameter or
strain/parameter derivative of the same scalar energy; no Jacobian or Hessian is
materialized. Periodic or particle-backend problems declare the reusable
layout's `search_cell` and `search_margin` on the execution plan.

`examples/atomistic_mace.py` labels strained periodic cells with a teacher MACE
through `energy_and_forces(..., compute_stress=True)` and fits a differently
initialized student on energy, force, and stress. Both models are random, so the
example demonstrates the workflow, not accuracy.

Every loader global batch is one accumulation window of the shared
accepted-update kernel, with `MODEL` root authority and one kernel objective per
declared objective; numerators and supports merge before normalization. Typed
lifecycle events are delivered through an explicit `IterationSession`;
observation sinks cannot alter training, while an `IterationHostControl` may stop
at an update or payload boundary.

A model, scale, cutoff, slot, objective, or component mismatch is refused before
any optimizer state exists. A window that reads a nonfinite label or produces a
nonfinite or capacity-failing evaluation ends the run with
`AtomisticStatus.NONFINITE` or `NEIGHBOR_OVERFLOW` at the last committed
boundary, so `result.potential` is always the last accepted state. A zero-support
window is a counted skip. A model selected by patience or host control retains
`STOPPED_EARLY`; neither case is reported as an ordinary maximum-step
completion.

`AtomisticTrainingResult` retains the current and selected trees (`potential`,
`best_potential`), the committed training-kernel state (parameters, optimizer and
plateau state, EMA target, any open window, root key, and cursors), the fitted
`AtomisticTrainingNormalization`, the loader cursor, per-update window values
and per-objective values, validation values and steps, failure and skip counts,
status, termination, and the problem, execution, policy, continuation,
reduction, capability, and checkpoint identities.

Continue by raising `maximum_steps` and passing `continuation=result`. The
continuation restores the committed state, cursors, selection state, and
histories exactly. A changed optimizer, objective, weighting, adaptation,
validation cadence, patience, delta, or selection policy is refused rather than
silently starting a different run. The graph execution plan may be re-planned,
for example to grow a capacity and replay a failed window; the loader buckets
and placement are bound by the cursor. Bitwise equality with the uninterrupted
run requires ordered streamed accumulation in every streamed plan the model
executes, such as `StreamedRelationPlan(accumulation="deterministic")` or
`"compensated"`, and a fixed device, runtime, and compiler profile. Host
iteration sessions are segment-local: resume with `session=None` or a fresh
session.

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
A `MACELESPotential` artifact also stores its prepared latent Ewald action: the
definition, the execution resources, and, for a periodic binding, the
lattice-family reciprocal mode set. Restore replays the definition and resources
through their constructors. Before generating any canonical mode, it admits the
declared mode layout against its budget and binds it to the already decoded mode
arrays. It then compares those arrays with the canonical half space in
fixed-size blocks rather than preparing them again. A stale definition, unit
factor, atom capacity, budget, or mode set is refused with bounded memory, and a
fresh process evaluates the same finite or periodic energy.
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

- `write_atomistic_training_restart(directory, result, problem, policy, *,
  source=None, licenses=(), reduction=None)` atomically publishes the artifact
  of the merged `result.potential`, the training kernel checkpoint (parameters,
  optimizer and plateau state, EMA target, any open accumulation window, root
  key, loader cursor, selection, best tree, and histories), and a
  `restart.json` digest receipt as one bundle. The receipt binds the model
  artifact, result, problem, execution, policy, continuation, reduction,
  capabilities, normalization, kernel, and loader-cursor identities. Under
  `policy.adaptation` the merged model is not an admissible restore template,
  because the kernel's frozen lane binds the adapted base; the bundle therefore
  also stores that base as `base.phydrax`. A failed publication preserves the
  previous resumable bundle. The training data recipe is not stored.
  `read_atomistic_training_restart(directory, problem, policy, *,
  components=None, reduction=None)` rebuilds against the same
  `AtomisticTrainingProblem`, restores from the model (or adapted base)
  artifact, and refuses mismatched receipts, artifacts, kernel states,
  adaptation, and receipt-free directory layouts. A host stop inside an
  accumulation window is resumable exactly. Resume with
  `fit_atomistic_potential(artifact.model, problem, policy, continuation=result)`
  and no (or a fresh) iteration session.
  `write_atomistic_training_checkpoint(path, result, problem, policy)` and
  `read_atomistic_training_checkpoint(path, potential, problem, policy)` are the
  kernel-only boundary. Normalization is refitted on restore and must
  reproduce its identity.
- `write_atomistic_restart(path, plan, state, sources=None, licenses=None)`
  bundles the native artifact of every learned term the prepared Hamiltonian
  executes, each term's binding (name, force group, periodic admission, bound
  conditioning with its default-use evidence, typed random key, observable
  request) and the dynamics state. `sources` and `licenses` are keyed by learned
  term name; unknown names refuse. Terms whose artifacts agree in full identity
  share one archived artifact. A fresh process restores the bindings with
  `read_atomistic_restart_models`, which maps each term name to an
  `AtomisticRestartModelBinding` (`artifact`, and `term` rebuilt over exactly
  `artifact.model`), rebuilds its dynamics, and resumes with
  `read_atomistic_restart`. A missing, additional, or swapped learned term, an
  altered model, conditioning, or key, a wrong source binding, system,
  neighborhood bundle plan or capacity levels, integrator, thermodynamic
  table, scope, or state layout refuses. A restart written after a capacity
  replacement records the bundle recipe with its grown capacity levels;
  `read_atomistic_checkpoint_resources(path).bind(hamiltonian)` rebinds a
  freshly prepared Hamiltonian to them before its dynamics are rebuilt.
  Compiled caches are not part of restart correctness.
  `examples/periodic_mace_dynamics.py` resumes from such a bundle and compares
  the result with the uninterrupted trajectory.

Restored states are always exact. Later steps reproduce the uninterrupted run
bitwise only when the executed graph plans select ordered streamed accumulation
and the resuming process uses the same device type, runtime versions, and a
fixed compiler profile. XLA GPU autotuning times candidate kernels in each
process and can select different tilings or emitters, so fresh-process bitwise
continuation on GPUs also requires autotuning disabled in both processes, for
example `XLA_FLAGS=--xla_gpu_autotune_level=0`. Otherwise continued values can
differ at roundoff, and a training problem whose labels are recomputed on the
accelerator in the new process can carry a different problem identity, which
`read_atomistic_training_restart` refuses.

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
local NPZ or an explicit URL plus mandatory SHA-256. For every seed it builds
training and validation sources and loaders from the same split (`--batch-size`
draws per update, optional `--payload-cases` accumulation) and uses the same
objectives, weights, optimizer, cutoff, capacity, feature width, interaction
count, and radial basis for PaiNN and NequIP. It records errors, equivariance
defects, compile/steady timings, host memory, model parameters, dense-candidate
and active-neighborhood work, accepted updates and skipped windows, predeclared
gates, per-model and paired summaries, tensor-product plan evidence, and
environment provenance. It prints an artifact only when run; no data or
benchmark result is bundled with PhydraX.
