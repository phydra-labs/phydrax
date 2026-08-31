# Finite-molecule atomistic learning

This guide covers PhydraX's finite, nonperiodic molecular energy-learning
workflow. It uses the existing material-particle and `GraphIR` substrates rather
than introducing another entity or graph system. The PaiNN/NequIP prediction and
training surface described here does not itself claim periodic execution, stress,
long-range electrostatics, or molecular-dynamics stability. Conservative
atomistic simulation is a separate prepared execution path documented in the
[atomistic dynamics guide](guides_atomistic_dynamics.md).

## Scale and atom identity are part of the input

Every structure carries an `AtomisticScaleContract`. Unit names and conversion
to a user-chosen reference system are content-addressed; PhydraX does not infer,
convert, or silently combine units.

```python
import phydrax as phx

scale = phx.atomistic.AtomisticScaleContract(
    "angstrom",
    "electronvolt",
    length_to_reference=1.0,
    energy_to_reference=1.0,
)
water = phx.atomistic.AtomicStructure(
    [8, 1, 1],
    [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
    [15.999, 1.008, 1.008],
    scale,
    particle_ids=[100, 101, 102],
)
```

`AtomicStructure` prepares a `ParticleSetPlan`/`ParticleDiscretization`, so
stable IDs, masses, and the active mask have the same semantics as other
material-particle methods. Active atomic numbers are positive. Atomic number
zero is reserved for inactive padding, and inactive entries must be zero.
`AtomisticBatch.from_structures` pads without changing an active atom's ID or
mass. All cases in one batch must have the same exact scale identity.

Cell and periodic-axis arrays can be preserved for provenance while parsing or
moving data. `PaiNNPotential` and `NequIPPotential` reject any such metadata.
Neither interprets a periodic structure as a free-space molecule.

## Case-isolated graphs and resource contracts

PaiNN and NequIP use the same directed dense candidate topology per case. No candidate joins two
molecules. Runtime displacement, distance, unit direction, node/edge masks, and
neighbor counts are stored in a canonical `GraphIR`. Coincident atoms have zero
direction rather than a nonfinite normalization.

Dense candidate storage is quadratic, so every realization uses an explicit
`AtomisticGraphExecutionPlan` with a `maximum_dense_atoms` guard.
`maximum_neighbors` is a separate runtime capacity contract on that execution
plan. The implementation evaluates every candidate and reports an overflow; it
never truncates, clips, repairs, or selects a partial neighbor set.
`AtomisticGraph.require_success` and direct energy evaluation fail closed.
`energy_and_forces` instead returns `valid=False`, status
`NEIGHBOR_OVERFLOW`, and non-trustworthy values as NaN so a batch pipeline can
retain typed failure evidence.

```python
execution = phx.atomistic.AtomisticGraphExecutionPlan(
    16,
    maximum_dense_atoms=32,
)
batch = phx.atomistic.AtomisticBatch.from_structure(water)
graph = phx.atomistic.realize_atomistic_graph(
    batch,
    execution,
    cutoff=5.0,
)
```

The dense path is the only atomistic neighborhood realization currently
exposed. There is consequently no cell-list parity claim.

## PaiNN scalar/vector interactions

`PaiNNPotential` embeds atomic number into invariant scalar features. Each
interaction combines a smooth sinusoidal radial basis and cosine cutoff with
scalar messages and Cartesian vector messages. Channel maps are native PhydraX
`Linear` layers using the parameter-transform contract; contractions use
`opt_einsum.contract`. Vector channels only undergo channel mixing, invariant
inner products, scalar gating, and multiplication by relative unit directions.
The total energy is a masked sum of invariant per-atom scalar readouts and is
therefore translation-, rotation-, and atom-permutation-invariant.

```python
import jax.random as jr

potential = phx.nn.atomistic.PaiNNPotential(
    scale,
    cutoff=5.0,
    feature_count=64,
    interaction_count=3,
    radial_basis_count=20,
    key=jr.key(0),
)
prediction = phx.atomistic.energy_and_forces(potential, batch, execution)
```

## Low-degree Cartesian NequIP interactions

`NequIPPotential` is a drop-in alternative under the same `AtomicStructure`,
`AtomisticBatch`, `energy_and_forces`, and `fit_atomistic_potential` contracts.
It embeds species into invariant scalar channels, forms scalar, vector, and
symmetric-traceless rank-two edge features, and applies weighted Cartesian O(3)
tensor products. Receiver aggregation, species-conditioned equivariant self
connections, and parity-safe gates update the node state. Only invariant scalar
channels enter the masked per-atom energy readout.

```python
nequip = phx.nn.atomistic.NequIPPotential(
    scale,
    cutoff=5.0,
    feature_count=32,
    interaction_count=3,
    radial_basis_count=20,
    key=jr.key(2),
)
nequip_prediction = phx.atomistic.energy_and_forces(nequip, batch, execution)
```

`O3TensorProductPlan` resolves every legal degree/parity instruction, the
canonical fully connected `uvw` multiplicity weights, component normalization,
parameter count, coefficient storage, scalar contraction work, resource limits,
and a content ID before `O3TensorProduct` prepares coefficients or weights.
Radial networks emit one coefficient for every actual tensor-product
instruction weight rather than one scalar per output block. Padded nodes and
edges are masked at embedding, edge-feature, message, aggregation, and readout
boundaries.

This implementation is independently derived in the existing Cartesian
scalar/pseudoscalar, vector/pseudovector, and symmetric-traceless
tensor/pseudotensor convention. Its declared scope is degrees zero through two
on finite nonperiodic molecules. It is not an arbitrary irreps API and makes no
claim for higher degrees, MACE or symmetric contraction, periodic stress,
long-range electrostatics, or molecular-dynamics stability.

The force is defined only as the negative position gradient of the same scalar
total-energy closure. Dense candidate indices and their topology identity stay
frozen during that derivative; the smooth cutoff controls interaction support.
There is no direct-force output and no stress output. `AtomisticPrediction`
contains energy, per-atom energy, forces, validity/status, overflow evidence,
maximum neighbor work, net force, center-of-mass net torque, scale, and
provenance. Net force and torque diagnose numerical equivariance defects; they
are not silently projected to zero.

`AtomisticPrecisionPolicy` separately declares coordinate, interaction,
reduction, and output dtypes. Prediction provenance records scale and precision,
the architecture identity, and a content fingerprint of the exact evaluated
parameter state. Training refreshes that state identity for both final and
selected-best potentials.

`PaiNNPotential(...)` and `fit_atomistic_potential(...)` return checkpointed
models. An external Equinox or Optax tree update changes numeric parameters but
necessarily preserves static metadata; it is unsupported for provenance-bearing
prediction until explicitly checkpointed:

```text
updated = phx.atomistic.checkpoint_atomistic_potential(updated)
prediction = phx.atomistic.energy_and_forces(updated, batch, execution)
```

The immutable checkpoint operation returns a new potential and is shared by the
abstract atomistic-potential contract so additional equivariant architectures
use the same provenance boundary.

## Energy, force, or joint training

Training is domain-specific; it does not add a generic trainer.
`PaiNNPotential` and `NequIPPotential` use the same
`AtomisticTrainingProblem`, which pairs native batches with optional molecular energies
and/or Cartesian force components. Masks explicitly select labeled cases and
components. The energy loss is mean squared error per atom. The force loss is
mean squared error per selected Cartesian component. Their weights are separate.
Loss scales can be supplied, or fitted from the training targets only: standard
deviation of per-atom training energy and RMS training-force component. A
validation split never contributes to those fitted values.

```text
problem = phx.atomistic.AtomisticTrainingProblem(
    training_batch,
    execution,
    training_energy=training_energy,
    training_forces=training_forces,
    validation_batch=validation_batch,
    validation_energy=validation_energy,
    validation_forces=validation_forces,
)
policy = phx.atomistic.AtomisticTrainingPolicy(
    maximum_steps=1_000,
    learning_rate=1e-3,
    energy_weight=1.0,
    force_weight=100.0,
    validation_every=10,
    patience=20,
)
result = phx.atomistic.fit_atomistic_potential(
    potential,
    problem,
    policy,
    key=jr.key(1),
)
```

The fit loop uses the shared `TrainingController` for the master key, ordered
callbacks, selection, patience, and progress. `AtomisticTrainingResult` retains
final and best potentials, optimizer state, key, fitted normalization, every
training component history, validation values and steps, progress, status, and
termination identity. Continue deterministically by raising the total step
ceiling and passing `continuation=result`; loss, optimizer, normalization, and
selection semantics must remain identical.

Nonfinite loss or gradient terminates with `AtomisticStatus.NONFINITE`. A model
selected by patience or a callback retains `STOPPED_EARLY`. Neither case is
reported as an ordinary maximum-step completion.

## Local rMD17 archives

`load_rmd17_npz` only reads a user-provided local archive. It accepts common
rMD17 field names for nuclear charge, coordinates, energy, force, and optional
original sample indices. The default declared scale is angstrom and
kilocalorie-per-mole; override it when an archive uses different units. The
parser performs no download and does not import a foreign atomistic framework.

`split_rmd17` makes deterministic, disjoint train/validation/test indices and
fingerprints the exact dataset, seed, and index arrays. The default sizes are
950/50/1000. `RMD17Dataset.take` returns a native `AtomisticBatch` and aligned
energy/force arrays.

The developer benchmark tool `tools/atomistic_rmd17_benchmarks.py` accepts a
local NPZ or an explicit URL plus mandatory SHA-256. For every seed it uses the
same split, optimizer/loss policy, cutoff, capacity, feature width, interaction
count, and radial basis for PaiNN and NequIP. It records errors, equivariance
defects, compile/steady timings, host memory, model parameters, dense-candidate
and active-neighborhood work, predeclared gates, per-model and paired summaries,
tensor-product plan evidence, and environment provenance. It prints an artifact
only when run; no data or benchmark result is bundled with PhydraX.
