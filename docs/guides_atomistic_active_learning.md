# Atomistic adaptive learning

The adaptive-learning layer is a host-side scientific transaction over existing
atomistic rollout, committee, provider, training, qualification, and lifecycle
contracts. It is not a scheduler and it never invokes an authoritative provider from
inside compiled dynamics.

## Dimensionless acquisition

`CommitteeAcquisitionScorePolicy` converts energy, maximum-force, and maximum-atom
committee deviations to dimensionless components using declared positive scales.
`AcquisitionAggregation.MAXIMUM` selects the largest normalized defect;
`EUCLIDEAN` uses their Euclidean norm. `AcquisitionPlan` requires this policy, so
quantities with unlike physical dimensions are never added directly.

Selection remains deterministic: eligibility is score-based, the first frame has the
largest score, and subsequent frames maximize descriptor distance with score and
source index as tie breakers. Every `AcquisitionRecord` retains all three normalized
components and the scoring-policy ID.

## Immutable labels

`label_atomistic_acquisitions` evaluates selected degree-of-freedom frames through one
`AbstractExternalAtomisticProvider`. Each `AtomisticLabelRecord` binds coordinates,
energy, forces, optional stress, provider, acquisition, system, topology, and units.
Only records carrying `successful=True` may enter `AtomisticLabelSet`.

A label set carries a canonical `phydrax.NumericRevision` of its ordered label IDs
(semantics: system, topology, and unit system) and a `lifecycle.RevisionLineage`.
An append creates a child revision whose lineage names the parent's canonical
`revision_id` and `lineage_id`; `label_set_id` binds both. Duplicate
configuration/provider pairs are rejected.
Training and validation membership (`AtomisticLabelSplit`) is stored on each label and
cannot be silently reshuffled by a campaign round.

External provider labels are authoritative interop. They do not satisfy a native-teacher
requirement; native teacher labeling and distillation are documented in
[Atomistic distillation](guides_atomistic_distillation.md).

## Labels as a canonical source

`AtomisticLabelSet.source(system, split=None)` exposes the labels (optionally one split)
as an immutable canonical `AtomisticStructureSource` of the shared training schema:

- one record per label, in label-set order; the record ID is the label ID and the
  independent-unit lineage is the exact configuration ID, so one configuration labeled
  by several providers is one unit;
- positions from the frame; for a periodic system, the frame cell vectors (the system
  cell when a frame carries none) and the system periodic axes. A finite system's frames
  cannot carry a cell;
- species (atomic numbers for element systems, atom-type IDs otherwise), masses in the
  system mass unit, and stable particle IDs from the prepared system, whose unit system
  must equal the label units;
- total energy and conservative forces, always available;
- a `stress` field whenever any record of the set carries provider stress: the native
  tensile Cauchy stress `(1/V) dE/d(strain)` in the system pressure unit
  (`AtomisticStressConvention.CAUCHY_TENSION_POSITIVE`, the
  `ExternalAtomisticEvaluation.stress` convention). Records without stress mark it
  unavailable; they never become zero targets. Every split therefore shares one schema.

`AtomisticLabelSet.training_problem(system, graph_execution, objectives, *, cutoff,
skin=0.0, batch_size=None, seed=0, weighting="literal", precision=None)` lowers the
`train` source to an epoch loader (`batch_size=None` is full batch) and the `validation`
source, when present, to a validation loader. Validation must be lineage-, identity- and
content-disjoint from training (`AtomisticSplitEvidence`); a configuration labeled in both
splits is refused. `objectives` bind the canonical `energy`, `forces` and `stress` slots;
the atomistic trainer remains the single implementation of the optimization, and graph
topologies are prepared per payload for `cutoff + skin`. Periodic or particle-backend
training plans declare their route-search envelope (`search_cell`, `search_margin`).

## One campaign round

`run_atomistic_campaign_round` performs:

1. runtime-resource coverage of every initial member (before any provider call);
2. normalized uncertainty/diversity selection;
3. authoritative provider evaluation;
4. immutable label append;
5. independent member retraining with `fit_atomistic_potential`;
6. conversion of selected models to learned-graph runtime programs;
7. committee construction;
8. caller-supplied physical qualification;
9. transactional promotion.

`AtomisticLearningCampaignPlan(system, provider, acquisition, graph_execution,
runtime_graph_execution, training, committee_reduction, *, objectives=...,
runtime_resources=..., skin=0.0, batch_size=None, seed=0, weighting="literal",
precision=None)` makes every choice explicit. `objectives` is a nonempty, uniquely named
sequence that may bind only the provider label slots `energy`, `forces` and, for a
periodic system, `stress`; a slot without an objective is simply not supervised. Loss
forms, reductions, scales and schedules live on the objectives, and `training` is the
`AtomisticTrainingPolicy` shared by every member. Every member is fitted on the same lowered
problem, whose cutoff is the largest member cutoff. Training uses `graph_execution`, which
need not be dense. Each trained member enters the candidate committee as a
`LearnedGraphPotentialTerm` program on the particle `runtime_graph_execution`, prepared with
the one `runtime_resources` bundle plan whose neighborhood requirements equal the member's;
members sharing a prepared bundle share one certified neighborhood state in committee
evaluation.

Promotion evidence is identity-bound. Each member's model manifest associates the label
set, the training problem (which binds the loaders' source mixtures, graph preparation,
objectives, weighting and precision), the training policy, and the potential's capability
identity. Campaign plan, training-problem and manifest identities differ from those
recorded before the streaming training cutover.

Promoted models are persisted with the pickle-free native model artifacts and training
restarts documented in [Atomistic learning and dynamics](api/atomistic.md).

A provider failure does not mutate the label revision. A failed member or failed
qualification preserves the previously promoted committee. An empty acquisition is a
successful no-op with explicit evidence.

Committee disagreement is an acquisition and trust diagnostic, not a calibrated
Bayesian posterior. Thresholds must be chosen from an immutable calibration set.
