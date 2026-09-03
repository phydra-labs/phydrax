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

A label-set append creates a new `lifecycle.NumericRevision` whose parent is the
previous content digest. Duplicate configuration/provider pairs are rejected.
Training and validation membership is stored on each label and cannot be silently
reshuffled by a campaign round.

`AtomisticLabelSet.training_problem` lowers the immutable labels to the existing dense
`AtomisticTrainingProblem`; the atomistic trainer remains the single implementation of
energy/force optimization.

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

Training uses a dense graph execution plan. Runtime committee programs require a
separate particle graph plan. The distinction is explicit in
`AtomisticLearningCampaignPlan`.

A provider failure does not mutate the label revision. A failed member or failed
qualification preserves the previously promoted committee. An empty acquisition is a
successful no-op with explicit evidence.

Committee disagreement is an acquisition and trust diagnostic, not a calibrated
Bayesian posterior. Thresholds must be chosen from an immutable calibration set.
