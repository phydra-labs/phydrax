# Native atomistic generation streams

Generation composes three native contracts in `phydrax.atomistic`:

- `AtomisticGenerationSupport` declares the proposal's chemistry, units, atom
  capacity, condition ports, and cell envelope. Fixed chemistry binds species,
  masks and masses. Variable composition uses an explicit element vocabulary and
  `AtomisticCompositionBounds`; it does not infer bonds, charge states,
  stereochemistry, or molecular identity. Such claims require a declared qualifier,
  such as `AtomisticChargeBalanceQualifier`.
- `AbstractPreparedAtomisticSampler` owns the numerical lane transition:
  `vacant`, `admit`, `advance`, `terminal`, and `emit`. One-shot samplers are terminal
  after admission; stateful samplers retain their complete method state. Samplers
  receive keys from their caller and derive role substreams through `role_keys`.
- `AtomisticGenerationStreamPlan` owns bounded attempts, pending results, and
  publication. It does not replace the sampler's scientific admission or numerical
  solver. The same scheduler runs a local stream (its own single participant) and a
  stream placed on an execution group; there is no second distributed scheduler.

## Identity, rejection, and continuation

Attempts have semantic ordinals. Keys are derived from the root key, campaign,
support, sampler, role, and ordinal; `atomistic_proposal_id` uses the corresponding
scientific identity and ordinal. Neither identity depends on a resident lane,
completion order, or publication timing. Numerical lane/chunk invariance remains a
contract of the selected sampler and execution route.

Terminal proposals wait in a bounded pending bank and publish in ordinal order.
Only a sink receipt advances the watermark. Rejected proposals retain their
`AtomisticProposalStatus` and filter evidence; rejection is not silent omission.
The target is the first requested number of accepted proposals in ordinal order.
Reaching it stops admission only: admitted lanes keep their complete method state
and pending results stay in the bank (`cursor.active` counts the in-flight lanes),
so `extend` continues them exactly and no drawn ordinal is discarded. An exhausted
attempt budget reports `BUDGET_EXHAUSTED` and an explicit shortfall once the last
budgeted ordinal has been admitted and no admitted lane remains in flight. Zero
target or zero budget performs no admission. Longer searches use explicit finite
continuation segments through `extend`.

The continuation is one array-bearing `AtomisticGenerationState`: root key data,
sampler lanes, pending bank, cursor (including `active` and the committed receipt
count `receipts`), and `receipt_digest`, a SHA-256 chain over committed receipts.
`write_checkpoint` and `read_checkpoint` persist a local stream through the typed
array-artifact owner. The artifact binds declared generation-support profiles,
units, the scientific sampler identities, the numeric revisions of every
parameter-bearing owner the stream executes (sampler members, the condition
source, and each condition transform), and `executables`, the canonical
fingerprint of the members' ordered typed `signature_facts` (lane layout,
capacities, precision). Those support profiles are not released scientific
qualifications. Pending and active lanes were conditioned and drawn under exactly
those values and layouts, so a changed stream, sampler, condition row, transform
coefficient, or executable layout is refused, and a continuation written with an
earlier state layout does not restore. A sampler identity is the scientific law
of its proposals and stays independent of executable facts such as the lane
count; the executable binding keeps a continuation from restoring into another
layout of the same law. Checkpoints and recipes whose recorded identities lack
the executable binding, or whose sampler identities still contain execution
resources, are refused; prepare the stream again from its constructor arguments
instead of carrying the old identities forward. A placed stream
persists through `publish_checkpoint` and `restore_checkpoint` instead (see
[Distributed quotas](#distributed-quotas)). After a crash between publication and
checkpoint, `reconcile` checks the sink's durable membership and advances
already-published results without publishing them again.

Semantic and numeric identity are separate. An `AtomisticConditionSchedule`'s
`source_id` binds its spec, policy, first ordinal, and row count; its row values
are its `numeric_revision`. An `AtomisticConditionTransform` binds its declared
`transform_id`, specs, and the semantic identity of its function in
`transform_id`; the function's parameters are its `numeric_revision`. Functions
must hold numeric parameters as visible dynamic PyTree leaves: an opaque function
whose closure cells, defaults, read module globals, partial arguments, or static
fields hide arrays of any dtype (including integer tables and typed PRNG keys), or
that contains an `ExplicitFreeze` holder, is refused, with
or without declared IDs. Functions are then identified by `callable_payload`: a
`StrictModule` or a plain stateless module-level function is content-addressed,
and any other callable (closure, lambda, nested function, partial, or foreign
callable object) must declare `function_semantic_id` and `function_numeric_id`.
A declared numeric ID is held as dynamic `IdentityWords`, so it binds the numeric
revision without entering executable structure. Changing only numeric values
keeps the stream identity and its compiled executables but refuses an exact
restore.

`publish`, `run`, and `reconcile` take the sink of the publication authority; a
local stream is its own authority and always supplies it. A sink that returns no
receipt is backpressure: nothing is committed and the watermark and every pending
result stay. `AtomisticMemoryGenerationSink` is an explicit in-process output
policy, not a durable store. Use `AtomisticStructureStoreGenerationSink` when
publication must survive process loss. Inside an `AtomisticStagePlan` campaign, the
campaign owns publication and receipts around the same sampler lane contract.

## Distributed quotas {#distributed-quotas}

`AtomisticGenerationPlacement(group, *, maximum_round_bytes,
maximum_replicated_bytes)` places one stream on a live `ExecutionGroup`, and
`plan.prepare(placement=placement)` binds it. Without a placement the stream is the
one-participant case of the same scheduler.

- **Participants.** Every member process of the group is one participant, in
  ascending process order. A participant's position is its group-local
  `ExecutionGroupRank` along the group's first mesh axis, never the global process
  index, so subgroups are placed exactly like whole jobs. Constructing a placement
  on a process outside the group is refused, so a nonparticipant never enters a
  stream collective. Each participant exchanges through its first group device;
  `placement_id` binds the group and those exchange devices.
- **One global window.** The stream keeps one ordinal window, one accepted target,
  one attempt budget (an absolute exclusive ordinal limit), and one publication
  order. Every participant holds the same replicated cursor, pending bank, root key,
  and receipt chain, plus its own sampler lanes.
- **One exchange per transition.** Each participant harvests its terminal lanes as
  bank rows. One fixed-shape all-gather (`JaxCollectiveProvider`, entered by every
  participant in the same order even when nothing moves) carries those rows, the
  free-lane masks, and a cursor, root-key, and receipt-chain consensus check; a
  mismatch raises `AtomisticGenerationDivergenceError`. Every participant inserts
  every row into its replicated bank. Each bank slot (`ordinal mod
  pending_capacity`) was reserved when its ordinal was admitted, inside the window
  `[watermark, watermark + pending_capacity)`, so the bounded bank is the credit.
- **Deterministic allocation.** Every participant derives the same allocation of
  the next ordinals to `(participant, member, lane)`: free lanes of each routed
  member are ordered participant-major, then by lane, and ordinals take them in
  order until one finds its member full, so admission stays contiguous. Each
  participant admits only its own picks.
- **RNG and identity.** Proposal keys and identities depend on campaign, support,
  sampler, ordinal, and role only. Participant positions order free lanes for
  placement and never enter keys, identities, or publication order. Numerical
  invariance across lane counts remains a contract of the selected sampler.
- **Capacities.** `pending_capacity` must cover every resident lane of every
  participant. The prepared stream reports `round_bytes` (gathered bytes of its
  largest exchange) and `replicated_bytes` (bank, cursor, root key, and receipt
  chain on every participant). Either above its placement bound raises
  `AtomisticGenerationCapacityError` at preparation, before any work; a capacity
  refusal is not a scientific rejection.
- **Publication authority.** Group-local rank 0 hosts the sink: it passes the sink
  to `publish`, `run`, and `reconcile`, and every other participant passes `None`
  (any other combination is refused). The authority's receipt (sink and durable
  identities) is agreed through one exchange, and every participant commits the
  same receipt into the chain.
- **Persistence.** `publish_checkpoint(repository, state, *, checkpoint_id,
  writer_id, ...)` is collective and called by every participant at the same
  transaction boundary. The replicated root key, bank, cursor, and receipt chain
  are written once; each participant's lanes, including in-flight lanes retained
  at the target, are its own block, written under its group-local rank as the
  lifecycle writer slot. After a group barrier the authority writes the single
  global commit; artifacts without it never restore. `restore_checkpoint(repository,
  checkpoint_id)` requires the same stream profile (including lanes and pending
  capacity), sampler, executable-layout, condition-source, and transform
  revisions, and placement.
  Any change, including another participant or lane count, is refused with
  `AtomisticGenerationCheckpointError`; it is not migrated. After a restore,
  `reconcile` on the authority's durable sink commits proposals that were
  published after the checkpoint.
- **Failure.** A participant that disappears leaves its peers blocked inside an
  exchange. The launcher fails the coupled job within its deadline, and the run
  restarts from its last global commit. No collective rollback after the loss of a
  participant is promised.

`tools/atomistic_generation_distributed_smoke.py` (`quota`, `checkpoint-kill`, and
`restart` scenarios) and `tests/integration/atomistic/test_distributed_generation.py`
produce the evidence for the closure obligation
`atomistic-c19-distributed-generation-quotas`. Until a run is observed and
retained, that obligation is an implemented, unqualified candidate. The default
forced-CPU (gloo) launch is functional evidence on one host only, including for the
one-device-per-participant exchange mesh. Placed generation across hosts is the
separate external gate `atomistic-c19-physical-multi-host-generation`.

```bash
python -m tools.atomistic_generation_distributed_smoke --scenario quota \
    --directory .tmp/gen-smoke/q --local-processes 2
python -m tools.atomistic_generation_distributed_smoke --scenario checkpoint-kill \
    --directory .tmp/gen-smoke/r
python -m tools.atomistic_generation_distributed_smoke --scenario restart \
    --directory .tmp/gen-smoke/r
```

## Compiled periodic samplers

`PreparedPeriodicDiffusionSampler` and `PreparedPeriodicFlowSampler` are one-shot
samplers for fixed or variable cells and explicitly admitted compositions. Each
prepares its transport once: native reverse-time VE diffusion through
`PreparedReverseDiffusion`, or a probability-flow `ContinuousTransport`.

Admission uses a stable compiled entry point per sampler and resident lane count.
Admitted lanes are compacted in lane order and solved in bounded `lane_chunk`
chunks. Vacant lanes perform no score evaluation. Padding in the last chunk repeats
an admitted lane and discards the extra result, so work is bounded by the declared
chunk width. `lane_chunk` is an execution choice, not part of `sampler_id`.

The caller score receives the torus-wrapped state, `atom_mask`, optional species,
and declared condition values through `ScoreContext`. Inactive fractional slots
have zero score. If the caller domain lacks `atom_mask`, the lifted score domain
adds the indicator cube. Variable composition draws species through the native
categorical reverse chain with lane-bound conditions, then validates conditions
against the decoded composition. Coordinate integration never substitutes for
chemistry qualification.

The species denoiser is part of the sampler's identity, not only the categorical
chain (whose identity covers schedule, shape, and terminal reference). It is
admitted and identified like a condition transform function (no hidden arrays;
opaque denoisers declare `species_predictor_semantic_id` and
`species_predictor_numeric_id`). Its semantic identity enters `sampler_id`, so a
different composition-generating function yields other proposal identities and
keys. Its parameters enter the sampler's `numeric_revision` beside the schedule
and terminal probabilities, so changed denoiser weights refuse an exact restore
while reusing the compiled admission.

Cells use explicit height, condition, and volume bounds. The
`AtomisticLogCholeskyCellChart` parameterizes positive-volume cells; admission
failures remain visible rather than being repaired. `FlatTorusManifold` in
`phydrax.metrix` distinguishes fractional-Euclidean shortest paths from the
physical cell metric. Componentwise fractional wrapping is not generally the
shortest Cartesian displacement in a triclinic cell; physical minimum-image
queries retain branch and cut-locus evidence.

## Interpretation and evidence

Generation pseudotime is not molecular-dynamics time. Generated proposals have no
implied likelihood, Boltzmann weight, pretrained accuracy, or physical stability.
Geometric filters and crystal contact relief do not establish those claims.
Source-weight rights, scientific references, and device/distribution qualification
remain separate from native implementation and restart correctness. Agreement
between a placed stream and a local stream is checked within one backend and
precision profile; no GPU-to-CPU bitwise equality is claimed.

See `examples/atomistic_generation_stream.py` for the stream API and
`benchmarks/atomistic_periodic_generation.py` for phase-separated preparation,
lowering, compilation, execution, and memory measurements. Each timed admission
mask retains its own first and final repeated solver/status evidence; vacant lanes
are excluded from acceptance counts. These observations are separate from the
later stream's acceptance statistics. Benchmark results apply only to the recorded
model, capacities, tolerance, runtime, and device tuple.
