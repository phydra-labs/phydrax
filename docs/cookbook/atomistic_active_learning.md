# Run one atomistic learning transaction

Build acquisition scores from dimensionless committee defects and keep validation labels
immutable.

```python
import phydrax as phx

score = phx.atomistic.CommitteeAcquisitionScorePolicy(0.05, 0.2, 0.05)
acquisition = phx.atomistic.AcquisitionPlan(
    32,
    score,
    minimum_score=1.0,
)
plan = phx.atomistic.AtomisticLearningCampaignPlan(
    system,
    authoritative_provider,
    acquisition,
    training_graph,
    particle_runtime_graph,
    training_policy,
    committee_reduction,
)
state = phx.atomistic.AtomisticLearningCampaignState(initial_labels)
result = phx.atomistic.run_atomistic_campaign_round(
    plan,
    state,
    candidate_frames,
    candidate_uncertainty,
    independently_initialized_models,
    member_keys,
    qualify_candidate_committee,
    descriptors=kinetic_or_structural_descriptors,
)
if not bool(result.successful):
    raise RuntimeError("Campaign round failed")
state = result.state
```

Provider failure never appends a label. Member or qualification failure advances the
label revision but preserves the previously promoted committee. Persist the returned
state and its label-set `RevisionLineage` at every accepted round boundary.

`training_graph` is any `AtomisticGraphExecutionPlan`; `particle_runtime_graph` must
use the particle backend. Stress returned by the authoritative provider is kept on each
label and lowered into the retraining problem's stress targets, weighted by the
training policy; labels without stress are masked out rather than given zero targets.
To inspect the supervision a round trains on, lower the labels yourself with an
explicit candidate cutoff:

```text
problem = result.state.labels.training_problem(
    system, training_graph, cutoff=member_cutoff, skin=0.0
)
```

Write promoted members with `phx.atomistic.write_atomistic_model_artifact` so a fresh
process restores them without the original Python objects.
