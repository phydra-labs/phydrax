# Native atomistic distillation

Distillation fits a native student to a native **teacher**. Teachers are native
Phydrax models frozen on purpose, not external evaluators: their labels are evaluated
with the same graph preparation, derivative contracts and Hessian bindings as
training. External labels remain honest interoperability
(`label_atomistic_acquisitions`) but are not native teachers.

## Teachers and label identity

`AtomisticTeacher(model, name=...)` is an explicit freeze. Its `frozen` view passes
every inexact teacher leaf through `stop_gradient`, so no enclosing derivative reaches
the teacher while coordinate and strain derivatives (forces, stress, Hessian actions)
are computed normally. `teacher_id` binds the architecture, force method, numeric
revision (`atomistic_model_revision`), the fixed/model-state content and the name.

`AtomisticLabelRequest(energy=, forces=, stress=, atom_energy=, embeddings=, hvp=,
hessian_policy=)` declares the stored teacher quantities. `embeddings` binds one
representation `(observable, port, O3IrrepLayout)`: `NODE_EMBEDDINGS` (atom level)
or `GRAPH_EMBEDDINGS` (structure level), the exact port (layer and representation)
and the packed irrep layout (axis identity). `hvp=AtomisticHessianProbe(kind, count=,
seed=)` stores coordinate Hessian-vector products together with their probes; probe
`k` of a record is drawn from a key addressed by the record's content digest and `k`,
never by batch position, draw or lane. `atomistic_label_id(teacher, request)` is the
label definition identity.

Probe values are immutable float64 host preparation with a fixed CPU JAX RNG
route, then stored beside their HVP labels. Accelerator labeling/training keeps
that host backend enabled (for example `JAX_PLATFORMS=cuda,cpu`, with CUDA first).
Restricting JAX to CUDA alone excludes the declared probe-generation backend;
the workflow does not silently redraw probes on another backend.

`label_atomistic_structures(teacher, loader, fields, request, graph_execution=,
cutoff=, store=)` evaluates the teacher natively (jitted over batches of
`AtomisticBatchPreparation`) and streams the labeled records through the canonical
`AtomisticStructureStoreWriter` `store` (the `atomistic-data` extra). It returns the
committed, lazily read `AtomisticStoredStructureSource` with these teacher fields:

| Slot | Level | Unit | Convention |
|---|---|---|---|
| `teacher_energy` | structure | energy | `total-energy` |
| `teacher_forces` | atom, (3,) | force | `conservative-force` |
| `teacher_stress` | structure, (3, 3), optional | energy/length^3 | stress convention |
| `teacher_atom_energy` | atom | energy | `teacher-atom-energy;teacher=<id>;complete=<bool>` |
| `teacher_embedding` | atom or structure, (P,) | 1 | `teacher-embedding;port=<id>;layout=<id>` |
| `hessian_probe` | atom, (count, 3), derived | length | `hessian-probe;probe=<id>` |
| `teacher_hvp` | atom, (count, 3) | force | `coordinate-hessian-vector-product` |

Every labeled record keeps the source lineage and receives the record ID
`label-<digest>` of (label definition, source record ID, source record digest), so
labels of different teacher revisions or definitions stay distinguishable and
labeling is idempotent. `require_atomistic_labels(labeled, teacher, request, source)`
refuses stale, reordered or foreign labels from record metadata alone.

Labeling is bounded end to end. The loader's epoch 0 must draw every record exactly
once in canonical order on one process (`draw="epoch"`, `shuffle=False`, unit
weights, every shard local); shuffled, repeated or partially local epochs are refused.
Only one global loader batch of labels, Hessian probes and re-read source rows is
resident; labeled records are appended whenever `store.layout.records_per_shard` are
pending, and the writer's `AtomisticStoreLimits` bound every shard, commit, read and the
retained record count. No call reads a whole source. The store is named
`teacher-labels-<label_id>` and lives as long as its directory: an interrupted run
(crash, refused commit) resumes after the committed prefix, which must be exactly the
expected records under the same schema; a complete store is returned unchanged; a store
of another label definition, source, order or with a retraction is refused and never
overwritten. A failed teacher evaluation raises `AtomisticTeacherLabelError` and
publishes nothing of its batch.

## Distillation objectives

E/F/S matching uses the P7 objectives on the teacher slots
(`AtomisticEnergyObjective("teacher_energy")`, `AtomisticForceObjective("teacher_forces")`,
`AtomisticStressObjective("teacher_stress")`). Atom-energy matching is a
decomposition-bound regularizer: `AtomisticAtomEnergyObjective("teacher_atom_energy",
decomposition=<the field convention>, complete=...)` refuses another teacher's gauge.

- `AtomisticEmbeddingObjective(student_spec, teacher_spec)` matches representations
  through a trainable `AtomisticEmbeddingProjector(student_layout, teacher_layout,
  key=)`, an `O3IrrepLinear` multiplicity-space map: scalars get a learned dense
  projector, nontrivial irreps an equivariant one; teacher irreps without a student
  irrep of equal degree and parity are refused. Pass the projector with
  `fit_atomistic_potential(..., components={"embedding": projector})`.
- `AtomisticHessianVectorObjective(probe, policy=)` matches stored teacher HVPs with
  the student's Hessian action under the admitted mixed third-order request
  `ATOMISTIC_HVP_TRAINING_REQUEST = DifferentiationRequest((INPUT, MODEL_PARAMETER),
  order=3)`. Models whose derivative contract does not admit it (for example
  almost-everywhere models without an explicit `RegularityPolicy`) are refused at
  admission. Strain HVPs have their own mixed request and are not served by it.
- `AtomisticBoltzmannObjective(groups, forward_weight=)` is a self-normalized
  finite-sample KL estimator: for each group, forward
  `sum wbar (u_s - u_p) + log sum w e^{-(u_s-u_q)} - log sum w e^{-(u_p-u_q)}` with
  `wbar ∝ w e^{-(u_p-u_q)}`, the reverse estimator with teacher and student exchanged,
  mixed by `forward_weight`. `u = beta (E + P V)` is the unbiased teacher/student
  reduced potential; `u_q` is the generator's authenticated density. It is a
  consistent but biased estimator, not the population KL; sampled coordinates are
  fixed data (no differentiation through sampling). Its support counts groups and its
  numerator sums per-group estimates, so each group must be complete in one payload.

Distribution samples are authenticated by the canonical sampling owner, never by
caller-supplied identifiers, densities or record-to-draw declarations. A generator is
either `AtomisticGeneratorState(result, k, qualification=, dynamics=, selection=None)`
(ONE fixed generating row of a committed `AtomisticMultistateSegmentResult`) or
`AtomisticGeneratorMixture(result, qualification=, dynamics=, selection=)`. Both admit
draws only through `AtomisticMultistateEquilibriumSamples` (unsuccessful segments,
adapting SAMS or bias-history draws, unauthenticated frozen histories and wrong-owner
bias evidence are refused), require the segment's
`AtomisticCanonicalSamplingQualification` of `dynamics` (with fixed row Hamiltonians),
and take `u_q(x_n)` from the owner's reduced potentials, which include each row's
frozen bias. The segment must have captured its draw configurations
(`AtomisticMultistateSegmentPlan(..., configurations=AtomisticDrawConfigurationCapture(
max_bytes=))`, an explicit bounded budget recorded inside the native scan; MBAR-only
segments stay without geometry output). `generator.samples`
(`AtomisticGeneratorSamples`) holds the exact unwrapped configuration, species and
cell of every admitted column on the qualified system. A single-row generator admits
the active draws of its row (with `selection` only on segments of at least two rows).
`−βU_old` is an unnormalized reduced energy: samples of several rows enter one group
only through the mixture, which resolves `selection` once to
`FreeEnergySelectionEvidence` on the segment's `ReducedPotentialDataset`, solves MBAR
with that same evidence and forms `u_mix = -log sum_k (N_k/N) exp(f_k - u_k)` from its
retained counts; burn-in or thinned columns carry no density and no sample.
Poor-overlap, disconnected, rank-deficient, insufficient or nonconverged evidence is
refused.

`AtomisticBoltzmannGroup(generator, beta=, composition=, pressure=None)` (or
`from_table(table, k, composition=, generator=)`, which refuses a biased target row and
another measure or reduced convention) binds the generator's measure and convention,
the target state, one chemistry (composition, never just an atom count) and exactly the
generator's admitted samples. `atomistic_generator_source(generator)` returns the
canonical unlabeled records of those samples (one per column, chain lineage); label
them with the teacher. `atomistic_boltzmann_source(labeled, groups, membership=,
columns=, store=)` proves every record's claim: its positions, species, masses,
particle IDs and (periodic) cell must equal the claimed column's captured
configuration bitwise, in the sampled system's units, so a relaxation frame, another
column's configuration or a perturbed geometry is refused. It also refuses excluded,
duplicated or omitted columns, records of another chemistry and another energy unit,
and writes membership, the authenticated generator density and `P V` through the
bounded store writer after a bounded proof pass.

## On-policy distillation

`AtomisticOnPolicyDistillationPlan` declares the teacher and request, the initial
student (and trainable components), an `AtomisticDistillationRollout` (system,
program template with the named student learned term, resources, integrator,
thermodynamic states, walkers, `cadence` steps per frame and `frames` per segment),
objectives, training policy, graph execution, cutoff, replay `capacity`, `admission` and
`replay` draw, the `AtomisticDistillationQualification(labels, source,
maximum_loss=, require_improvement=)` (teacher labels of a fixed, lineage-disjoint
source; the plan refuses labels that `require_atomistic_labels` does not prove to be
exactly its own teacher and request's labels of `source`) and
the `drive` policy. Training always continues the latest trained candidate; `drive`
selects the model that generates the next rollout (`"promoted"` waits for promotion,
`"candidate"` drives with the latest trained candidate).

`run_atomistic_distillation_segment(plan, state, *, directory, stop_after=None,
session=None)` executes the committed phases from `state.phase`; `stop_after` returns
after the named phase commits:

1. **ROLLOUT** runs (or resumes from its campaign store) one native campaign of the
   driving student; every frame publishes durably into the `AtomisticExperienceBuffer`
   (idempotent by semantic key). The student is rebound by rebuilding its learned term
   and re-preparing program and dynamics; no hook mutates a prepared runtime. Each
   walker carries its complete `AtomisticDynamicsContinuation` (kinematics, cell,
   thermostat/barostat controllers, step, time, RNG, images, term states, force cache,
   neighborhoods) from frame to frame, so observation cadence never changes the
   dynamics; `PreparedAtomisticCampaign.terminal_packets` returns it as the next
   continuation. A walker whose final outcome is not COMPLETED fails the phase.
2. **LABEL** attaches teacher labels to every retained unlabeled frame (never twice).
   The attachment is durable before the TRAIN restart; after an interruption between
   the two, LABEL resumes from `AtomisticExperienceBuffer.label_successor`, which
   authenticates exactly that attachment (direct successor, recorded change and label
   content digests); any other replay change is refused as stale.
3. **TRAIN** fits the latest trained candidate on the immutable replay revision.
   `AtomisticReplayDraw(age_repeats)` repeats a frame of policy age `a`
   `age_repeats[min(a, len(age_repeats) - 1)]` times per epoch, so the final entry
   covers every older age class and a zero entry excludes that class. The actual draw
   probabilities are recorded in the report.
4. **QUALIFY** evaluates the candidate's selected view on the qualification source.
   `AtomisticQualificationOutcome.reason` is `"qualified"`, `"failed-evaluation"`,
   `"above-maximum-loss"` or `"no-improvement"`.
5. **PROMOTE** publishes a qualified candidate as a native model artifact and promotes it;
   a rejected candidate never replaces the promoted student. When the next driving
   student differs, every walker continuation is first moved to its operation by the
   owner's Hamiltonian transition (configuration, momenta, time, step, RNG and
   controllers kept, forces re-evaluated, lineage recorded) while the previous driving
   student is still held; no per-walker model copy is kept.

After every committed transition `write_atomistic_distillation_restart` publishes the
state (walker continuations with their operation binding and lineage, cursors,
revisions, policy history, qualification, pending publication, students as native
model artifacts and, while committed, the training owner's restart members).
`read_atomistic_distillation_restart(directory, plan)` resumes in a fresh process
against a rebuilt plan, authenticates every walker continuation through the driving
student's operation and refuses stale or foreign replay revisions. Failed
phases return the last committed state with `report.failure`;
`AtomisticDistillationSegmentResult.successful` means no phase failed, not that a
candidate was promoted.

Use `stop_after="qualify"` to retain `state.qualification` before promotion clears
the segment-local decision. `examples/atomistic_distillation.py` demonstrates the
LABEL restart, captures the qualification reason and values, and then completes
PROMOTE. Inspect both the qualification decision and `report.promoted`.
The report's `phase_seconds` are host transition timings captured before restart
publication, not independently synchronized device-kernel timings.

`AtomisticExperienceAdmission(outcomes=..., require_evaluation=True,
maximum_force=None)` declares which campaign outputs enter replay; refused frames keep a
durable `AtomisticRefusalRecord` reason and FIFO eviction keeps an
`AtomisticEvictionRecord`. Experience replay is unrelated to `AtomisticReplayPolicy`
(deterministic trajectory recomputation). Restarts and publications need
artifact-registered student models (for example MACE).

## Evidence and gates

A random native teacher demonstrates the mechanics of labeling, training,
qualification and promotion. It is not evidence of pretrained or chemical accuracy, of
teacher-to-student fidelity on real chemistry, or of any source framework's
distillation results. A real distillation additionally needs:

- an admitted teacher whose weights, source configuration and redistribution or
  derivative-use rights are recorded for the intended use;
- teacher and student derivative contracts that admit every requested label and
  objective, including the mixed third-order request for HVP matching on the selected
  native route; there is no accelerated or tabulated third-order route;
- a qualification source and threshold chosen from independent data;
- observed evidence for each device, precision and distribution tuple actually used.
