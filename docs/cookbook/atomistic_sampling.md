# Biased sampling, window exchange and frozen-history analysis

The blocks below form one script. They follow a pair-distance coordinate of a two-atom
Lennard-Jones dimer through the native enhanced-sampling path:

1. A Hamiltonian with bias terms.
2. Well-tempered metadynamics, including terminal exhaustion and capacity growth.
3. A bias-aware window exchange that is still adapting.
4. Freeze-to-analysis MBAR with an unbiased target row whose coupling is zero.
5. ABF, both force-only and integrated.

All runs are deliberately tiny. They show the contracts and the evidence each step
records. They are not convergence evidence.

## Bias terms in the scalar Hamiltonian

Biases are ordinary terms of `AtomisticPotentialProgram`. Forces, constraint projection,
Monte Carlo cell trials and reduced-potential cross-evaluation therefore all see one
definition. The umbrella center is a control, and the hill history is read-only term
state. Both are bound by the term's declared slot `name`; the exact `term_id` authenticates
the bound content.

```python executable
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


phx.config.configure(phx.config.RuntimeConfig(enable_x64=True))
atomistic = phx.atomistic
sampling = phx.atomistic.sampling

system = atomistic.AtomisticSystemPlan(
    [0, 1],
    [1, 1],
    [1.0, 1.0],
    atomistic.AtomisticUnitSystem.reduced(),
    atom_type_ids=[0, 0],
).prepare()
positions = jnp.asarray([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])
measure = atomistic.AtomisticPhaseSpaceMeasurePlan(system)

distance = sampling.CollectiveVariableProgram(
    (
        sampling.CollectiveVariablePlan(
            sampling.CollectiveVariableKind.DISTANCE, [0, 1]
        ).prepare(system),
    ),
    names=("distance",),
)
umbrella = sampling.AtomisticBiasTerm(
    distance,
    sampling.HarmonicBiasForm([1.2], 20.0),
    name="distance-umbrella",
    force_group=1,
)
history = sampling.AtomisticBiasHistoryPlan(
    1,
    height=0.05,
    widths=0.05,
    capacity=2,
    interval=2,
    bias_factor=8.0,
    temperature=1.0,
)
metadynamics = sampling.AtomisticBiasTerm(
    distance,
    sampling.HillHistoryBiasForm(history),
    name="distance-metadynamics",
    force_group=1,
)
lennard_jones = atomistic.LennardJonesPotential([0.2], [1.0], 2.5)
program = atomistic.AtomisticPotentialProgram([lennard_jones, umbrella, metadynamics])
(pairs,) = program.requirements.neighborhoods
# Finite system: a bounded dense image source searches an explicit nonperiodic frame.
bundle = atomistic.AtomisticNeighborhoodBundlePlan(
    (pairs,),
    (
        atomistic.AtomisticNeighborhoodSource(
            "lennard-jones-pairs",
            atomistic.AtomisticSiteDomain.PHYSICAL_ATOMS,
            (pairs,),
            atomistic.AtomisticDenseImageSearch(maximum_dense_routes=4),
            phx.discretization.ParticleImageCapacityLadder(
                (
                    phx.discretization.ParticleImageCapacity(
                        maximum_particles_per_cell=2,
                        maximum_edges=2,
                        maximum_degree=1,
                        maximum_images=1,
                        maximum_case_edges=2,
                    ),
                )
            ),
            skin=0.3,
        ),
    ),
    (
        atomistic.AtomisticRoleCapacityLadder(
            pairs,
            (atomistic.AtomisticRoleCapacity(maximum_routes=1, maximum_degree=1),),
        ),
    ),
    frame=phx.discretization.PeriodicCell(
        16.0 * jnp.eye(3), origin=jnp.full((3,), -8.0), periodic_axes=(False,) * 3
    ),
)
potential = program.prepare(system, neighborhood_bundle=bundle)
# Canonical rows need a canonical thermostat; velocity Verlet is NVE only.
dynamics = atomistic.AtomisticDynamicsPlan(
    system, potential, atomistic.BAOABLangevinPlan(1.0e-3, 1.0)
).prepare()
assert dynamics.conservative_force_definition
```

## Well-tempered metadynamics and terminal exhaustion

`PreparedBiasedDynamics` adds no integrator. Each committed iteration runs in this order:

1. Propagate with the current Hamiltonian.
2. Deposit any hill that is due, with its height computed against the pre-event history.
3. Re-prime the forces at the committed configuration.

The jump in energy is booked as `history_work`. Force group 0 (Lennard-Jones) is the
physical Hamiltonian, and the bias terms are excluded from it.

Terminal storage refuses the deposit that would overflow. The refused iteration commits
nothing. Its `HISTORY_EXHAUSTED` reason is latched, and every later step is refused until
`grow_bias_history_capacity` rebinds a larger buffer. The scientific identity is
unchanged: capacity is a resource.

```python executable
canonical = atomistic.AtomisticThermodynamicStatePlan(
    measure, ensemble="nvt", temperature=1.0, state_id="metadynamics"
)
walker_table = canonical.prepare(dynamics)
walker_runtime = sampling.PreparedBiasedDynamics(
    dynamics, walker_table, physical_force_groups=(0,)
)
walker = walker_runtime.initialize(
    dynamics.initialize_state(
        positions,
        walker_table,
        velocity=jnp.zeros_like(positions),
        key=jax.random.key(0),
    )
)
outcomes = []
for _ in range(6):
    evaluation = walker_runtime.step(walker)
    outcomes.append(bool(evaluation.successful))
    walker = evaluation.accepted_state
exhausted = int(sampling.BiasedStepReason.HISTORY_EXHAUSTED)
# Hills are due at committed steps 2 and 4; the third deposit (step 6) does not fit.
assert outcomes == [True] * 5 + [False]
assert int(evaluation.reasons) & exhausted
assert int(walker.dynamics.step_index) == 5
assert int(walker.deposition_events) == 2 and float(walker.history_work) > 0.0
latched = walker_runtime.step(walker)
assert int(latched.reasons) & exhausted
assert int(latched.accepted_state.dynamics.step_index) == 5

walker_runtime, walker = sampling.grow_bias_history_capacity(
    walker_runtime, walker, metadynamics.name, 16, (canonical,)
)
for _ in range(3):
    evaluation = walker_runtime.step(walker)
    assert bool(evaluation.successful)
    walker = evaluation.accepted_state
assert int(walker.dynamics.term_states[metadynamics.name].history.written) == 4
dynamics = walker_runtime.dynamics  # The rebound Hamiltonian owns the grown history.

with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "walker.npz"
    written = sampling.write_biased_dynamics_checkpoint(path, walker_runtime, walker)
    restored = sampling.read_biased_dynamics_checkpoint(path, walker_runtime, walker)
    assert restored.payload_id == written.payload_id
```

## Freezing a history

`FrozenBiasHistory` fingerprints every kernel array and counter. Its `declaration` value,
`"frozen:<digest>"`, is the only form in which a thermodynamic row can name fixed history
content. `metadynamics_profile` is a diagnostic. It records its storage, ownership,
temperature and convergence assumptions, and it is never an equilibrium
`FreeEnergyResult`.

```python executable
frozen = walker_runtime.freeze(walker)[metadynamics.name]
assert frozen.declaration == f"frozen:{frozen.digest}"
assert not frozen.exhausted and not frozen.fifo_overwritten
metadynamics_estimate = sampling.metadynamics_profile(
    frozen,
    distance.metrics,
    (jnp.linspace(1.0, 1.4, 21),),
    sampled_temperature=1.0,
)
assert metadynamics_estimate.formula_admitted
print("metadynamics assumptions", metadynamics_estimate.assumptions)
```

## Bias-aware window exchange with an adaptive history

Each row binds umbrella centers with `AtomisticBiasRowBinding`. Every history term must be
declared identically in every row. The rows' bias identities enter their potential IDs,
so cross evaluation includes each row's actual bias.

An `"adaptive"` history deposits at committed iterations, using every walker's
pre-event values. Draws taken while it adapts are recorded per draw, and none of them is
an equilibrium sample.

```python executable
def window(center, declaration, state_id, coupling=1.0):
    return atomistic.AtomisticThermodynamicStatePlan(
        measure,
        ensemble="nvt",
        temperature=1.0,
        state_id=state_id,
        bias=sampling.AtomisticBiasRowBinding(
            {umbrella.name: [center]},
            couplings={umbrella.name: coupling, metadynamics.name: coupling},
            histories={metadynamics.name: declaration},
        ),
    )


def multistate(table, run_id, exchange=None):
    return sampling.AtomisticMultistatePlan(
        table,
        [0, 1],
        qualification=sampling.AtomisticCanonicalSamplingQualification(
            dynamics,
            table,
            "finite-step-baoab-cookbook-qualification",
            sampling_exact=False,
            sampling_bias_bound=1.0e-2,
        ),
        exchange=exchange,
        run_id=run_id,
    ).prepare(dynamics)


def lanes(table, seed):
    return tuple(
        dynamics.initialize_state(
            walker.dynamics.kinematics.positions,
            table,
            state_index=index,
            momentum=walker.dynamics.kinematics.momenta,
            key=jax.random.key(seed + index),
            term_states=walker.dynamics.term_states,
        )
        for index in range(2)
    )


exchange_table = atomistic.PreparedThermodynamicStateTable(
    dynamics,
    (window(1.1, "adaptive", "window-1.1"), window(1.3, "adaptive", "window-1.3")),
)
exchange_runtime = multistate(
    exchange_table, "cookbook-adaptive-windows", sampling.AtomisticReplicaExchangePlan(1)
)
replicas = exchange_runtime.initialize(
    lanes(exchange_table, 10), [0, 1], jax.random.key(20)
)
adaptive = sampling.AtomisticMultistateSegmentPlan(
    exchange_runtime, 4, 0, 0, exchange_runtime.initial_continuation_id
).run(replicas)
assert bool(adaptive.successful)
assert adaptive.bias_history_declarations == ("adaptive",)
assert bool(jnp.all(adaptive.bias_adapting))
assert not bool(jnp.any(adaptive.sample_active))
# Iterations 2 and 4 each insert one kernel per walker into the shared history.
assert int(jnp.sum(adaptive.bias_deposited)) == 2
assert int(adaptive.bias_history_revision[-1, 0]) == 8
assert len(adaptive.bias_history_digests()) == 1
```

## Freeze-to-analysis MBAR with an unbiased target row

The analysis table declares the frozen digest in every row. The third row keeps the same
terms at zero coupling, so it is the unbiased target. No replica samples it, and MBAR
estimates it through cross evaluation.

Initialization authenticates each lane's bound history content against the declaration.
A row carrying a matching-looking label for other content is refused. The checkpoint
manifest and the UQ conversion authenticate the content again.

Every row of an analysis table is one fixed Hamiltonian, evaluated at every sample
column. A shared history has one owner. A state-owned or frozen walker-owned history
evaluates the owner key its row declares in `AtomisticBiasRowBinding(owner_keys=...)`,
so a frozen multi-walker history is analyzed through one row per state and owner. An
adaptive walker-owned history evaluates each replica's own kernels and is never an
analysis row. `AtomisticMultistateEquilibriumSamples` is the single admission of
segment draws for every reduced-potential consumer, including one-state tables. It
refuses unsuccessful segments, adapting draws, unauthenticated frozen content, and any
draw whose evaluated history owner differs from its row's declared owner.
`reduced_potential_dataset_from_multistate` builds the MBAR dataset from it.
A segment plan given `configurations=AtomisticDrawConfigurationCapture(max_bytes=...)`
also records each committed draw's evaluated configuration (wrapped positions, image
counts, species and cell); a footprint above `max_bytes` is refused when the plan is
built, and the policy enters the segment and checkpoint identities.

```python executable
forged_table = atomistic.PreparedThermodynamicStateTable(
    dynamics,
    (
        window(1.1, "frozen:" + "0" * 64, "forged-1.1"),
        window(1.3, "frozen:" + "0" * 64, "forged-1.3"),
    ),
)
forged_runtime = multistate(forged_table, "cookbook-forged-label")
try:
    forged_runtime.initialize(lanes(forged_table, 50), [0, 1], jax.random.key(60))
except ValueError as error:
    print("refused:", error)
else:
    raise AssertionError("a frozen label without its content must be refused")

analysis_table = atomistic.PreparedThermodynamicStateTable(
    dynamics,
    (
        window(1.1, frozen.declaration, "window-1.1"),
        window(1.3, frozen.declaration, "window-1.3"),
        window(1.1, frozen.declaration, "unbiased", coupling=0.0),
    ),
)
analysis_runtime = multistate(analysis_table, "cookbook-frozen-analysis")
replicas = analysis_runtime.initialize(
    lanes(analysis_table, 30), [0, 1], jax.random.key(40)
)
analysis = sampling.AtomisticMultistateSegmentPlan(
    analysis_runtime, 8, 0, 0, analysis_runtime.initial_continuation_id
).run(replicas)
assert bool(analysis.successful)
assert analysis.bias_history_digests() == (frozen.digest,)
assert not bool(jnp.any(analysis.bias_deposited))
assert not bool(jnp.any(analysis.bias_adapting))
assert bool(jnp.all(analysis.sample_active))

checkpoint_plan = sampling.AtomisticMultistateCheckpointPlan(analysis_runtime, 8)
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "analysis.phx"
    written = sampling.write_atomistic_multistate_checkpoint(
        path, checkpoint_plan, analysis
    )
    restored = sampling.read_atomistic_multistate_checkpoint(
        path, checkpoint_plan, replicas
    )
    assert restored.payload_id == written.payload_id
    assert restored.segment.bias_history_digests() == (frozen.digest,)

samples = sampling.AtomisticMultistateEquilibriumSamples(analysis)
assert analysis_runtime.qualification.fixed_row_hamiltonians
assert samples.bias_history_digests == (frozen.digest,)
assert samples.reduced_potentials.shape == (3, 8 * 2)
dataset = phx.uq.reduced_potential_dataset_from_multistate(analysis)
np.testing.assert_array_equal(
    np.asarray(dataset.values), np.asarray(samples.reduced_potentials)
)
estimate = phx.uq.multistate_bennett_acceptance_ratio(
    dataset, phx.uq.FreeEnergySelectionPlan(block_length=1), reference_state=2
)
assert bool(jnp.all(jnp.isfinite(estimate.free_energies)))
assert not bool(estimate.successful)  # The declared finite-step kernel is approximate.
print("window free energies vs unbiased target", np.asarray(estimate.free_energies))
```

## ABF: force-only port and integrated profile

Force-only ABF is a typed nonconservative force port. It has no energy, so its runtime
reports `conservative_force_definition=False` and refuses energy-based consumers:

- the canonical-sampling qualification, so no multistate runtime or analysis, even
  for a single state;
- multi-state thermodynamic tables;
- Monte Carlo volume moves;
- integrator-owned cell methods and RESPA splitting;
- switching work.

The estimator below is fed known physical forces of `U = a r`. Each `ABFSample` carries
the native Gram-solve evidence in `dual_solve` (an `ABFLinearSolveEvidence`);
`fixman_solve` is `None` because the closure has no constraints. The integrated profile
is compared with the independent radial reference `A(r) = a r - 2 kT log r`.
`IntegratedProfileBiasForm` turns that frozen profile into an ordinary conservative term.
The profile itself is a diagnostic `AdaptiveProfileResult`, not an equilibrium free
energy.

```python executable
abf = sampling.AtomisticABFPlan(
    distance,
    minimum=0.9,
    maximum=1.5,
    bins=12,
    temperature=1.0,
    min_samples=1,
    full_samples=1,
).prepare(system)
slope = 0.5
centers = np.asarray(abf.plan.centers()[0])
abf_state = abf.initialize()
for r in centers:
    pair = jnp.asarray([[0.0, 0.0, 0.0], [float(r), 0.0, 0.0]])
    forces = jnp.asarray([[slope, 0.0, 0.0], [-slope, 0.0, 0.0]])
    sample = abf.sample(sampling.CollectiveVariableGeometry(pair), forces)
    assert bool(sample.regular) and bool(sample.dual_solve.successful)
    assert sample.fixman_solve is None
    abf_state = abf.accumulate(abf_state, sample)
assert int(abf_state.irregular) == 0 and int(abf_state.out_of_range) == 0
profile = sampling.abf_profile(abf, abf_state)
assert profile.formula_admitted and profile.connected
reference = slope * centers - 2.0 * np.log(centers)
np.testing.assert_allclose(
    np.asarray(profile.free_energy) - np.asarray(profile.free_energy)[0],
    reference - reference[0],
    atol=5.0e-3,
)

port = sampling.AtomisticABFForcePort(abf, name="distance-abf")
physical = atomistic.AtomisticPotentialProgram([lennard_jones]).prepare(
    system, neighborhood_bundle=bundle
)
abf_dynamics = atomistic.AtomisticDynamicsPlan(
    system, physical, atomistic.BAOABLangevinPlan(1.0e-3, 1.0), force_ports=(port,)
).prepare()
assert not abf_dynamics.conservative_force_definition
abf_table = atomistic.AtomisticThermodynamicStatePlan(
    measure, ensemble="nvt", temperature=1.0, state_id="abf"
).prepare(abf_dynamics)
try:
    sampling.AtomisticCanonicalSamplingQualification(
        abf_dynamics,
        abf_table,
        "force-only-abf-cookbook",
        sampling_exact=False,
        sampling_bias_bound=1.0,
    )
except ValueError as error:
    print("refused:", error)
else:
    raise AssertionError("a force-only runtime has no canonical qualification")
abf_runtime = sampling.PreparedBiasedDynamics(
    abf_dynamics, abf_table, physical_force_groups=(0,)
)
abf_walker = abf_runtime.initialize(
    abf_dynamics.initialize_state(
        positions, abf_table, velocity=jnp.zeros_like(positions), key=jax.random.key(5)
    )
)
for _ in range(2):
    evaluation = abf_runtime.step(abf_walker)
    assert bool(evaluation.successful)
    abf_walker = evaluation.accepted_state
assert int(abf_walker.abf_samples) == 2

integrated = sampling.AtomisticBiasTerm(
    distance,
    sampling.IntegratedProfileBiasForm(profile),
    name="integrated-abf",
    force_group=1,
)
integrated_dynamics = atomistic.AtomisticDynamicsPlan(
    system,
    atomistic.AtomisticPotentialProgram([lennard_jones, integrated]).prepare(
        system, neighborhood_bundle=bundle
    ),
    atomistic.BAOABLangevinPlan(1.0e-3, 1.0),
).prepare()
assert integrated_dynamics.conservative_force_definition
prepared_integrated = integrated.prepare(system)
energy, _, successful = prepared_integrated.bias(
    sampling.CollectiveVariableGeometry(positions), None
)
# V_bias = -A with the profile gauge A = 0 at the first supported node.
expected = -(slope * 1.2 - 2.0 * np.log(1.2) - reference[0])
assert bool(successful)
np.testing.assert_allclose(float(energy), expected, atol=5.0e-3)
```

A segment with a missing or nonfinite required cross energy is unsuccessful. A frozen
declaration whose content does not match is refused at initialization, checkpoint
restore and UQ conversion. Draws taken under an adaptive history never become
equilibrium samples.
