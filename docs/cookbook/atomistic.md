# Train a finite-molecule PaiNN or low-degree NequIP potential

This recipe starts from a local rMD17 NPZ. It never downloads data and never
assumes units from array shape. The archive must contain nuclear charges,
coordinates, molecular energies, and Cartesian forces under one of the field
names accepted by `load_rmd17_npz`.

```text
from pathlib import Path

import jax.random as jr
import phydrax as phx

scale = phx.atomistic.AtomisticScaleContract(
    "angstrom", "kilocalorie_per_mole"
)
dataset = phx.atomistic.load_rmd17_npz(
    Path("/absolute/path/to/rmd17_aspirin.npz"),
    scale=scale,
)
split = phx.atomistic.split_rmd17(
    dataset,
    train_size=950,
    validation_size=50,
    test_size=1000,
    seed=0,
)
train_batch, train_energy, train_forces = dataset.take(split.train_indices)
validation_batch, validation_energy, validation_forces = dataset.take(
    split.validation_indices
)
test_batch, test_energy, test_forces = dataset.take(split.test_indices)
```

Inspect the molecule size before declaring dense resources. The guard is an
acceptance boundary, not a hint: a batch whose padded atom capacity exceeds it
is rejected. The neighbor limit is also fail-closed and is never implemented by
truncation.

```text
atom_capacity = train_batch.atom_capacity
execution = phx.atomistic.AtomisticGraphExecutionPlan(
    32,
    maximum_dense_atoms=atom_capacity,
)
potential = phx.nn.atomistic.PaiNNPotential(
    scale,
    cutoff=5.0,
    feature_count=128,
    interaction_count=3,
    radial_basis_count=20,
    key=jr.key(10),
)
```

To use degree-zero-through-two Cartesian NequIP without changing the graph,
prediction, training, or result path, replace only the model construction:

```text
potential = phx.nn.atomistic.NequIPPotential(
    scale,
    cutoff=5.0,
    feature_count=32,
    interaction_count=3,
    radial_basis_count=20,
    key=jr.key(10),
)
```

NequIP resolves and resource-checks its legal tensor-product instructions before
allocating layer coefficients and radial outputs. Its radial map has one output
for every multiplicity weight on every legal instruction. It remains a finite,
nonperiodic research model with degree at most two; this recipe does not imply
high-degree irreps, MACE-style contraction, stress, or periodic support.

Construct a typed joint problem. The default fitted scales use only training
energy and forces. The validation values are used for model selection, not
normalization.

```text
problem = phx.atomistic.AtomisticTrainingProblem(
    train_batch,
    execution,
    training_energy=train_energy,
    training_forces=train_forces,
    validation_batch=validation_batch,
    validation_energy=validation_energy,
    validation_forces=validation_forces,
)
policy = phx.atomistic.AtomisticTrainingPolicy(
    maximum_steps=500,
    learning_rate=1e-3,
    energy_weight=1.0,
    force_weight=100.0,
    validation_every=10,
    patience=20,
    min_delta=1e-6,
)
result = phx.atomistic.fit_atomistic_potential(
    potential,
    problem,
    policy,
    key=jr.key(11),
)
selected = result.best_potential
```

For energy-only training, omit the force targets and set `force_weight=0.0`.
For force-only training, omit molecular energies and set `energy_weight=0.0`.
At least one available target kind must have positive weight.

Evaluate the test split through the conservative prediction surface:

```text
prediction = phx.atomistic.energy_and_forces(selected, test_batch, execution)
if not bool(prediction.valid.all()):
    raise RuntimeError("Test prediction failed neighborhood or finite checks")

atom_count = test_batch.atom_counts
energy_error_per_atom = (prediction.energy - test_energy) / atom_count
force_error = prediction.forces - test_forces
```

`prediction.forces` is the negative derivative of `prediction.energy`; it is not
a separate learned head. Keep `prediction.provenance`, `split.split_id`, the
fitted `result.normalization`, and the resource settings with reported metrics.
Do not report stress, periodic behavior, long-range accuracy, or molecular-
dynamics stability from this finite nonperiodic workflow.

To continue the exact optimizer and selection state to a higher total step
ceiling:

```text
continued_policy = phx.atomistic.AtomisticTrainingPolicy(
    maximum_steps=1_000,
    learning_rate=1e-3,
    energy_weight=1.0,
    force_weight=100.0,
    validation_every=10,
    patience=20,
    min_delta=1e-6,
)
continued = phx.atomistic.fit_atomistic_potential(
    potential,
    problem,
    continued_policy,
    continuation=result,
)
```

Only `maximum_steps` may change across that continuation. A changed optimizer,
loss scale or weight, validation cadence, patience, delta, or selection policy
is rejected rather than silently starting a different run.
