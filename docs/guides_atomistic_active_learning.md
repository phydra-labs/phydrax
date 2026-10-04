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
Training and validation membership is stored on each label and cannot be silently
reshuffled by a campaign round.

`AtomisticLabelSet.training_problem(system, graph_execution, cutoff=..., skin=0.0)`
lowers the immutable labels to `AtomisticTrainingProblem` energy, force, and stress
supervision; the atomistic trainer remains the single implementation of the
optimization. Label units must equal the prepared system's unit system. Periodic records
keep their frame cell vectors. Provider stress labels are retained as tensile stress in
the system pressure unit (energy per cubic length of the system scale), the
`ExternalAtomisticEvaluation.stress` convention; records without stress stay
unsupervised through the stress mask rather than becoming zero targets. Each split
freezes its candidate graph topology for `cutoff + skin`.

## One campaign round

`run_atomistic_campaign_round` performs:

1. normalized uncertainty/diversity selection;
2. authoritative provider evaluation;
3. immutable label append;
4. independent member retraining;
5. conversion of selected models to particle-graph runtime programs;
6. committee construction;
7. caller-supplied physical qualification;
8. transactional promotion.

Training uses the campaign's `graph_execution` plan, which need not be dense; each round
freezes the training topology with the largest cutoff among the member potentials.
Runtime committee programs require a separate particle graph plan. The distinction is
explicit in `AtomisticLearningCampaignPlan`.

Promotion evidence is identity-bound. Each member's model manifest associates the label
set, the training problem (which binds the graph-execution plan, scale contract, and each
split's frozen topology and labels), the training policy, and the potential's capability
identity. Continuing a member requires the same concrete family, configuration,
capability identity, and training problem. Because frozen topologies, stress labels and
masks, and capability identities now enter these records, training-problem and campaign
manifest identities differ from those recorded by earlier versions.

Promoted models are persisted with the pickle-free native model artifacts and training
restarts documented in [Atomistic learning and dynamics](api/atomistic.md).

A provider failure does not mutate the label revision. A failed member or failed
qualification preserves the previously promoted committee. An empty acquisition is a
successful no-op with explicit evidence.

Committee disagreement is an acquisition and trust diagnostic, not a calibrated
Bayesian posterior. Thresholds must be chosen from an immutable calibration set.
