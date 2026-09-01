# Resumable trajectory output and rerun

Write accepted frames with a reporter, reopen the H5MD file, and rescore it with the same
prepared potential.

```python
from pathlib import Path
import phydrax as phx

path = Path("trajectory.h5")
sink = phx.atomistic.interchange.H5MDTrajectoryPlan(path)
reporter = phx.atomistic.AtomisticReporterPlan(sink, stride=10)

with sink.open(append=True) as writer:
    if int(state.step_index) % reporter.stride == 0:
        writer.write(reporter.frame(dynamics, state))

rerun = phx.atomistic.AtomisticRerunPlan(
    sink,
    potential,
    neighborhood,
    lambda_values=(0.0, 0.5, 1.0),
).run()
assert bool(rerun.successful)
```

`append=True` resumes at the committed frame boundary; it does not infer simulation state.
Resume dynamics from its atomistic checkpoint, then append frames whose system, topology,
and unit-system identities match the existing stream.

For analysis selections, convert an MDAnalysis selection once into an
`AtomisticSelectionPlan` and store the stable selected IDs. Do not execute selection strings
inside compiled dynamics.
