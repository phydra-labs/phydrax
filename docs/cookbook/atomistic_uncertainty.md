# Committee uncertainty

Committee members must share a prepared system and neighborhood contract.

```python
import phydrax as phx

committee = phx.atomistic.CommitteeAtomisticPotential(
    (potential_a, potential_b, potential_c),
    phx.atomistic.CommitteeReductionPolicy(
        0.5,
        1.0,
        0.2,
        policy=phx.atomistic.OODPolicy.REJECT,
    ),
)
evaluation = committee.evaluate(state.kinematics.positions, state.neighborhood)
if not bool(evaluation.successful):
    raise RuntimeError("committee member evaluation failed")

if bool(evaluation.uncertainty.out_of_domain):
    raise RuntimeError("trajectory segment rejected by committee policy")
```

Use `ConservativeUncertaintyBlend` when uncertainty must alter dynamics. It constructs a
single scalar energy whose gradient defines the force. For active learning, create
acquisition candidates only from accepted frames, attach the full evidence object, and run
deterministic diversity selection before invoking an authoritative labeling method.
