# Atomistic training data and streamed fitting

Atomistic training reads immutable structure sources through a deterministic
loader. One global batch of the loader is one accumulation window of the shared
accepted-update training kernel; full-batch training is the ordinary one-batch
loader, not a second trainer. The [atomistic learning guide](guides_atomistic.md)
covers the models, graph layouts and E/F/S objectives that consume these
batches.

## Structure sources and the structure store

`AtomisticStructureStoreSchema(levels, fields)` declares cardinality levels and
typed fields. Exactly one `"structure"` level (one row per record) and one
`"atom"` level exist; `"edge"`, `"group"`, `"segment"` and bounded `"product"`
levels (for example per-case Hessian blocks, with a required
`maximum_rows_per_structure`) may be added. Every `AtomisticStoreField(name,
level, dtype, *, kind, role, convention, axes=(), unit=None,
extensivity="not-applicable", availability="required", index=None)` binds:

- its exact stored dtype (no ingest cast) and trailing `Axis` identities;
- its value kind (`"continuous"`, `"integer"`, `"categorical"`, `"mask"`,
  `"index"`) and role (`"geometry"`, `"identity"`, `"condition"`, `"label"`,
  `"derived"`);
- an explicit unit and extensivity for continuous physical values, and an
  optional physical unit for integer quantities such as elementary-charge counts;
- per-row availability for `"optional"` fields, whose unavailable rows are
  canonical zeros, never targets.

A field is an index only through an explicit `AtomisticStoreIndex(target,
sentinel=None)`; integral storage alone does not make it one. Indices are stored
structure-local and rebased only on a `read(..., index_basis="batch-local")`.

`AtomisticStructureRecord(schema, *, record_id, lineage_id, counts, stable_ids,
values, available=None)` is one structure with its record identity,
independent-unit lineage, unique int64 stable particle IDs and content digest.
`AtomisticStructureSource(schema, records)` is the immutable in-memory source.
Reads preserve the requested order and duplicates; the source identity binds the
ordered record content independently of memory or disk layout.

The canonical on-disk store uses Zarr v3 through the optional `atomistic-data`
extra:

```text
limits = phx.atomistic.AtomisticStoreLimits(
    max_commit_bytes=1 << 24,
    max_member_bytes=1 << 28,
    max_shard_bytes=1 << 30,
    max_shard_members=4096,
    max_chunk_bytes=1 << 26,
    max_index_bytes=1 << 24,
    max_decoded_bytes=1 << 30,
    max_read_bytes=1 << 30,
    max_records=1_000_000,
)
writer = phx.atomistic.AtomisticStructureStoreWriter(
    "water-store",
    layout=phx.atomistic.AtomisticStoreLayout(
        records_per_shard=1024, target_chunk_bytes=1 << 20, compression="zstd"
    ),
    limits=limits,
)
head = writer.create(source.schema, name="water")
head = writer.append(head, source)
stored = phx.atomistic.open_atomistic_structure_store("water-store", limits=limits)
```

Writes seal immutable content-addressed shards before one exclusive commit per
revision is published, so a reader always sees a complete prior or new revision;
a stale or concurrent writer receives an `AtomisticStoreError` with reason
`"conflict"`. `retract(head, record_ids, reason=...)` and `compact(head)` publish
new lineage revisions and never rewrite sealed shards. Every limit is admitted
from commit metadata before any member is decoded. Chunking, shard size and
compression are execution metadata; they do not change record or source
identity. External or older stores enter only through explicit checked
interchange; there is no automatic repair.

## Canonical E/F/S sources

`atomistic_training_schema(scale, *, mass_unit, species_kind=..., periodic=False,
energy=True, forces=True, stress=False, stress_convention=...)` is the canonical
schema of the energy/force/stress trainer: `positions`, `species` and `masses`,
`cell` and `periodic` for periodic data, and the optional labels with explicit
units, axes and conventions.

`atomistic_training_source(batch, *, mass_unit, energy=None, forces=None,
stress=None, energy_available=None, force_available=None,
stress_available=None, record_ids=None, lineage_ids=None,
stress_convention=...)` ingests a fixed-capacity `AtomisticBatch` as one record
per case:

- energies `(case,)`, forces `(case, atom, 3)` and stress `(case, 3, 3)` in the
  batch scale, where stress is the tensile-positive `(1/V) dE/d strain`;
- availability per case, per atom and per case respectively. There is no
  per-component availability; stress components are selected on the objective;
- an available nonfinite label stays available and invalidates the update that
  reads it; it is never treated as missing;
- `record_ids` default to the batch structure IDs and `lineage_ids` to the
  record IDs, so by default every case is its own independent unit.

Stress requires cells and periodic axes. Stored labels keep their dtype;
float64 labels require JAX x64.

## Mixtures, loaders and workload buckets

`AtomisticSourceMixture(components, *, draw, objective, seed, shuffle=None,
draws_per_epoch=None)` binds the canonical order of
`AtomisticMixtureComponent(source, weight=...)` values:

- `draw="epoch"` uses integer weights as repeats per complete epoch (shuffled by
  default); `draw="replacement"` draws a declared `draws_per_epoch` with
  replacement from the normalized weights;
- `objective="sampled"` keeps unit case weights, while
  `"importance-corrected"` reweights draws to a uniform target over distinct
  records and requires a positive draw probability for every record;
- actual record probabilities and case weights are recorded; a repeated draw
  has its own draw ID but keeps its source record identity.

`atomistic_training_loader(*sources, batch_size=None, stream="training",
seed=0, shuffle=None, weights=None, max_bytes=1 << 34)` builds an epoch mixture,
binds every canonical field to the slot of the same name, and declares one
bucket that admits a whole global batch. `batch_size=None` is the full-batch
policy: every epoch draw is in one window.

To accumulate a window over several smaller payloads, keep the mixture and
bindings and declare explicit `AtomisticWorkloadBucket`s:

```text
canonical = phx.atomistic.atomistic_training_loader(water, hydrogen, batch_size=4, weights=(1, 2))
atom_level = canonical.mixture.schema.atom_level
rows = canonical.buckets.buckets[0].capacity(atom_level).rows
bucket = phx.atomistic.AtomisticWorkloadBucket(
    "molecules",
    cases=2,
    levels=(phx.atomistic.AtomisticLevelCapacity(atom_level, layout="per-case", rows=rows),),
    max_bytes=1 << 24,
)
training = phx.atomistic.AtomisticStructureLoader(
    canonical.mixture,
    canonical.bindings,
    phx.atomistic.AtomisticWorkloadBuckets((bucket,)),
    batch_size=4,
    stream="training",
)
```

A bucket declares its case count, one `AtomisticLevelCapacity(level,
layout="per-case" | "packed", rows=...)` for every counted level of the schema
(so atom-only, edge-only and combined budgets are explicit) and a byte bound.
Each draw enters the first bucket that admits its record. A record that no
bucket admits is refused with `AtomisticLoaderError` before any read; it is
never dropped silently. Padding repeats real rows with false validity and zero
case weight; it never invents a sample.

`remainder="short"` (default) keeps a short final batch; `"drop"` drops it.
`AtomisticFieldBinding` maps a store field onto a slot with its level, value
kind, role, convention, axes, unit and dtype. Integer physical quantities retain
their declared unit and exact integer values; only continuous bindings permit
explicit `convert_from` unit conversion. Categorical, mask and index bindings
remain unitless. A cached edge field binds its exact `derivation_id`.

## Placement, staging and cursors

`AtomisticLoaderPlacement(rank=None)` shards every global batch over the
group-local data shards of one execution-group rank. The default is one
single-device shard. Global shard `g` is the data-axis coordinate of the
execution group; this process addresses its contiguous `local_shards`. The
placement identity binds only the rank-independent layout, so every member
process shares one loader and cursor identity. For data-parallel training use
`AtomisticLoaderPlacement(distribution.process_rank)` of the
`AtomisticTrainingDistributionPlan` described in the
[distributed guide](guides_atomistic_distributed_execution.md#data-parallel-and-data-x-domain-training).

`AtomisticLoaderStaging(host_prefetch=0, target="host", device_ahead=0,
max_device_bytes=None)` is execution-only and outside every identity. Host
prefetch applies only when every source is background-read safe; device
staging requires an explicit byte bound covering `device_ahead + 1` payloads
and, when sharded, an execution group. A producer thread is not a claim of
overlapping device transfer.

`AtomisticLoaderCursor` binds source content, mixture, bindings, stream and
batch size (`semantics_id`), the bucket plan and placement (`structure_id`), and
the epoch, batch and in-window payload position. Resume validates the cursor
against the loader and refuses source drift or a changed bucket plan or
placement; it never skips a row count in a possibly reordered source. Random
per-draw keys (`AtomisticLoaderPayload.draw_keys`) are addressed by draw and
epoch, never by batch slot or placement.

Host provenance (`AtomisticPayloadProvenance`: source, record, lineage and
digest identities) travels beside the numerical payload. The static compile
signature contains only the bucket layout, so one executable per bucket and
graph layout serves every payload.

## Train, validation and test separation

`AtomisticSplitEvidence` refuses splits that share declared independent-unit
lineage, a record identity or a record-content digest. It covers declared
lineage and exact identities only; matching counts or group labels do not prove
scientific independence, so choose lineage IDs that name the actual independent
units (for example a trajectory or a parent configuration).

## Problem, batch preparation and objectives

```text
problem = phx.atomistic.AtomisticTrainingProblem(
    training,
    execution,
    (
        phx.atomistic.AtomisticEnergyObjective(),
        phx.atomistic.AtomisticForceObjective(),
        phx.atomistic.AtomisticStressObjective(),
    ),
    scale=units.scale,
    cutoff=5.0,
    fields=phx.atomistic.AtomisticTrainingFields(cell="cell", periodic="periodic"),
    validation=phx.atomistic.atomistic_training_loader(held_out, stream="validation"),
)
```

`AtomisticTrainingProblem(training, graph_execution, objectives, *, scale,
cutoff, skin=0.0, fields=None, validation=None, weighting="literal",
precision=None)` binds the loader, one declared objective per genuine target,
and an optional validation loader that must bind the same slots and pass
`AtomisticSplitEvidence`. Without a validation loader, validation evaluates one
training epoch. `AtomisticTrainingFields(positions="positions",
species="species", masses="masses", cell=None, periodic=None,
species_kind=..., conditioning=None, conditions=None)` names the geometry slots;
periodic data bind `cell` and `periodic` together, and condition slots bind an
explicit `AtomisticConditioningSpec`.

`problem.preparation` (an `AtomisticBatchPreparation`) prepares each payload on
the host: one `AtomisticGraphLayout` per bucket capacity and coordinate dtype,
then `layout.prepare(...)` into the `AtomisticGraphPayload` of an
`AtomisticTrainingBatch`. The graph layout and topology are never rebuilt inside
an update. Periodic or particle-backend problems therefore declare the reusable
layout's route-search envelope on the plan, `AtomisticGraphExecutionPlan(...,
search_cell=..., search_margin=...)`.

`problem_id` binds the data semantics, objectives, weighting and precision.
`execution_id` binds the graph execution plan and loader layouts. An in-process
continuation may re-plan the graph execution (for example a grown image
capacity); the loader bucket plan and placement are bound by the cursor and
must be unchanged.

### Objectives and reductions

Every objective returns an additive numerator and its support. Window numerators
and supports merge across payloads (and across data shards) before
normalization; unequal payload means are never averaged.

| Objective | Reductions | Support |
|---|---|---|
| `AtomisticEnergyObjective(slot="energy")` | `"per-atom-structure"` (default), `"total-structure"`, `"per-atom-atom-weighted"` | `Σ w_s` (atom-weighted: `Σ w_s N_s`) |
| `AtomisticForceObjective(slot="forces")` | `"component"` (default), `"atom"`, `"structure"` (graph-balanced) | per available component, per available atom, per structure with an available atom |
| `AtomisticStressObjective(slot="stress", components=...)` | `"component"` (default), `"structure"` with a `(3, 3)` component mask | selected components of available admissible cases |
| `AtomisticPropertyObjective(observable, port, slot, *, level, unit)` | `"component"`, `"structure"` | per available row; `"structure"` balances atom-level observables per case |
| `AtomisticAtomEnergyObjective(slot, *, decomposition, complete)` | none | per available atom of the declared decomposition |

`AtomisticResidualLoss("squared" | "absolute" | "huber" | "euclidean",
delta=None)` sets the residual form. Huber needs a positive `delta` (`r²/2`
inside, `delta (|r| - delta/2)` outside); `"euclidean"` is a vector norm
admitted only by the force `"atom"` and `"structure"` reductions. Residuals are
divided by `scale`; `scale=None` fits it from one training epoch only (RMS, or
the per-atom energy standard deviation). A fit without available labels, with
nonfinite labels, or below `policy.normalization_floor` is refused rather than
clamped. Property objectives refuse energy, forces, stress, atomic energies and
Hessian observables, which have dedicated conservative objectives. Atomic
energies are a decomposition-bound regularizer: their declared decomposition and
completeness must match the model.

`AtomisticObjectiveSchedule.constant(v)`, `.linear(a, b, steps=...)`,
`.cosine(a, b, steps=...)` and `.piecewise(boundaries, values)` are evaluated at
the committed-update count: fixed inside a window, never advanced by skipped or
rejected windows. `weighting="literal"` uses the weights as declared;
`"normalized"` divides by their sum at the same update and needs a schedule
whose minimum is positive.

`fit_atomistic_reference_energies(loader, *, species_slot, energy_slot, species,
rank_policy=None, split="training", head=None, convention="total-energy")` fits
per-species reference energies by native rank-revealing least squares over the
composition Gram matrix. It uses finite available energies of epoch zero
(nonfinite energies are excluded from this fit) and records rank, Gram singular
values, residual RMS, sources, split, head and energy convention. A
rank-deficient composition returns the minimum-norm solution unless the rank
policy requires full rank. Pass it as `AtomisticEnergyObjective(reference=fit)`.

`AtomisticTrainingPrecision(loss_dtype="float64", accumulation_dtype="float64")`
separates label storage, model compute, residual and accumulation dtypes; a
lower loss dtype is an explicit choice, and geometry derivatives never inherit
it.

Padding cases have zero weight and are inert. A failed model or graph evaluation
of any valid case in a window, whether a capacity overflow or a nonfinite
result, invalidates the whole window. A window with zero support is a committed
skip: it advances the loader cursor but no optimizer, schedule, plateau or EMA
state, so the next supported window still sees the same committed update step.

## Fitting and continuation

`fit_atomistic_potential(potential, problem, policy, *, key=..., session=None,
continuation=None, components=None, reduction=None)` uses the shared kernel:

- `AtomisticTrainingPolicy(maximum_steps, learning_rate=1e-3, optimizer=None,
  optimizer_id=None, clip_global_norm=None, adaptation=None, target=None,
  evaluation="current", plateau=None, selection_weights=None,
  validation_every=1, patience=None, min_delta=0.0, select_best=True,
  normalization_floor=1e-12, staging=None)`; a custom optimizer needs a stable
  `optimizer_id`;
- `adaptation=AtomisticParameterSelection(leaf_paths=...)` or
  `subtree_paths=...` trains only the selected leaves.
  `AtomisticIrrepLowRankAdaptation(sites, rank=...)` trains low-rank factors
  inside the multiplicity spaces of `O3IrrepLinear` maps
  (`atomistic_irrep_linear_sites(model)`), so the adapted model stays
  equivariant;
- `target=phx.nn.operator.training.ExponentialMovingAverageTargetPolicy(...)`
  with `evaluation="target"` validates and selects the EMA view.
  `plateau=AtomisticPlateauPolicy(...)` scales updates after stale validations
  as a committed rule-state transition.

`AtomisticTrainingResult` holds `trained_tree` and `best_tree` (with
`potential` and `best_potential`), the committed `training_state`,
`normalization`, the loader `cursor`, `training_values` and
`training_objectives` (updates × objectives), `validation_values`,
`validation_objectives` and `validation_steps`, `status`, `termination`,
`skipped_windows`, `unsupported_windows`, `capacity_failures` and
`scientific_failures`. A failed window ends the run with `NONFINITE` or
`NEIGHBOR_OVERFLOW` at the last committed boundary (state and cursor), so a
continuation replays it exactly. A host stop inside a window keeps the open
window.

`unsupported_windows` counts the consecutive skips since the last accepted
update (or the start of the run); continuations and checkpoints carry it. When
those skips cover one complete loader epoch, the training stream has offered no
supported window and the run stops at that epoch boundary with status
`STOPPED_EARLY`, termination `"unsupported_training_epoch"` and
`successful == False`: at the latest at the end of the first complete epoch
after the last accepted update. A supported window anywhere in an epoch keeps
training, including after unsupported windows that cross an epoch boundary.
Optimizer, schedule, plateau and EMA state stay exactly at the last accepted
update. Continuing such a result over the same problem stops again without
reading data; supply labelled data through a new problem instead.

Continue with a raised `maximum_steps` and `continuation=result`;
`write_atomistic_training_checkpoint(path, result, problem, policy)` and
`read_atomistic_training_checkpoint(path, potential, problem, policy)` persist
the kernel state for a fresh process, and the
[restart bundle](guides_atomistic.md#native-model-artifacts-and-restarts) adds
the native model artifact. A continuation equals the uninterrupted run bitwise
only with ordered streamed accumulation
(`StreamedRelationPlan(accumulation="deterministic")` or `"compensated"`) and a
fixed device, runtime and compiler profile.

`examples/atomistic_streaming_training.py` mixes two sources (the hydrogen
source drawn twice per epoch), packs every four-draw window as two-case
payloads, fits reference energies, schedules the force weight with a cosine,
clips the merged gradient, selects the EMA view on a lineage-disjoint
validation loader, and continues a checkpoint written after eight updates to
sixteen. Its teacher and student are random PaiNN initializations: it
demonstrates mechanics, not accuracy. `benchmarks/atomistic_training_streaming.py`
separates read, preparation and topology, compilation, first and warm updates,
repeated-layout compile counts, atoms, edges, padding and memory.
