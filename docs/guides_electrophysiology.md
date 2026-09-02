# Electrophysiology

The electrophysiology application provides a plan/prepare/state/evaluation/result architecture for compartmental cells, membrane mechanisms, ion concentrations, stochastic channel populations, and fixed-capacity multicell synapses. Host planning performs topology and capacity validation. Prepared runtimes contain fixed-shape JAX arrays and stable identities; compiled transitions never resize a state or mutate Python containers.

## Units and signs

Compiled kernels use an explicit unit contract:

| Quantity | Kernel unit |
|---|---|
| time | ms |
| voltage | mV |
| current | nA |
| conductance | µS (`uS`) |
| capacitance | nF |
| length | µm (`um`) |
| concentration | mM |
| temperature | K |

`convert_quantity(value, from_unit, to_unit)` rejects dimensionally incompatible conversions. Membrane and synaptic current is **outward positive**. Injected current and current-clamp amplitude are **inward positive**. Thus an affine membrane current is

```text
I_out(V) = conductance_uS * V_mV + current_offset_nA
```

and a conductance synapse has `current_offset_nA = -conductance_uS * reversal_mV`. Exact voltage clamps report the inward clamp current required by the physical capacitance, axial, membrane, and stimulus balance.

## Morphology planning

A `CompartmentSpec` is a cylindrical isopotential segment with a stable string ID, optional stable parent ID, length, diameter, membrane-capacitance density, and axial resistivity. `CellMorphologyPlan` requires exactly one root, a connected acyclic parent relation, unique stable IDs, and contiguous optional `BranchSpec` paths.

```python
from phydrax.applications import electrophysiology as ep

morphology = ep.CellMorphologyPlan(
    "pyramidal-cell",
    (
        ep.CompartmentSpec("soma", None, 20.0, 20.0),
        ep.CompartmentSpec("trunk", "soma", 80.0, 4.0),
        ep.CompartmentSpec("left", "trunk", 120.0, 2.0),
        ep.CompartmentSpec("right", "trunk", 120.0, 2.0),
    ),
    branches=(
        ep.BranchSpec("left-branch", ("soma", "trunk", "left")),
        ep.BranchSpec("right-branch", ("soma", "trunk", "right")),
    ),
).prepare()
```

Preparation computes membrane areas, capacitances, interface axial conductances, the Kirchhoff Laplacian, and postorder/back-substitution schedules. The prepared arrays are fixed by compartment count. `tree_elimination_solve` uses those schedules, while cable integration uses the PhydraX linear algebra runtime.

## Membrane programs and cable solves

`MembraneMechanism` defines steady-state initialization, exact gate updates, and outward-current affine coefficients. `MembraneProgram` evaluates mechanisms in deterministic order with three fixed gate lanes per mechanism. Built-in mechanisms are:

- `PassiveLeak`: ohmic leak;
- `HodgkinHuxleyNaK`: classic squid fast-sodium and delayed-rectifier potassium gates;
- `SodiumPotassiumPump`: ion-nonlinear, voltage-independent electrogenic current.

The pump is included exactly once in `current_offset_nA` and repeated in `nonlinear_current_nA` only as routing evidence. `MechanismStatus.NONLINEAR_ROUTED` records that path. Invalid/nonfinite ion concentrations fail the cable transition closed.

```python
program = ep.MembraneProgram(
    (
        ep.PassiveLeak(0.3, -65.0),
        ep.HodgkinHuxleyNaK(),
    )
)
solver = ep.CableSolverPlan(
    0.025,
    scheme="crank-nicolson",
).prepare(morphology, program)
state = ep.initialize_cable_state(
    solver,
    voltage_mV=[-65.0, -65.0, -65.0, -65.0],
)
result = ep.step_cable(solver, state, ep.zero_cable_inputs(solver))
```

Backward Euler and Crank–Nicolson assemble the exact theta-method system for frozen affine coefficients. Solves use `phydrax.linalg` and report absolute/relative linear residual, per-compartment Kirchhoff residual, global charge-balance residual, exact clamp current, finiteness, nonlinear routing, and a fail-closed bitwise status. `differentiable_dense_solve` delegates to the native mathematical implicit-derivative contract, preserving both forward-mode JVPs and reverse-mode VJPs. Gate dynamics use the exact affine exponential update rather than forward Euler, and nonfinite updated gates reject the entire step before state commit.

Differentiation is supported for fixed morphology, mechanism order, clamp activity, relation occupancy, event sequence, and state capacities. Discrete topology/event selection is not differentiated. Differentiate a fixed event realization, or introduce an explicit estimator outside this application.

## Stimulation, recording, and replay

`CurrentClamp` is a rectangular inward current. `VoltageClamp` is an exact rectangular Dirichlet command. `ElectrophysiologyProtocol.prepare` resolves their stable compartment IDs and a `RecordingPlan` into device indices. `RecordingState` has fixed sample capacity and records `CAPACITY_EXCEEDED` instead of resizing. A rejected cable step produces `REJECTED_CABLE_STEP` and does not advance the recording count or mark a duplicate sample valid.

```python
protocol = ep.ElectrophysiologyProtocol(
    ep.RecordingPlan(("soma", "left"), sample_capacity=4000),
    current_clamps=(ep.CurrentClamp("step", "soma", 0.25, 5.0, 20.0),),
).prepare(solver)
experiment = ep.initialize_experiment(protocol, state)
run = ep.run_experiment(protocol, experiment, 1000)
checkpoint = ep.checkpoint_experiment(protocol, run.state)
continued = ep.replay_experiment(protocol, checkpoint, 100)
```

Checkpoints are host-created and content-addressed over the complete cable and recording state plus prepared protocol identity. Restore rejects provenance or content mismatches. Replaying a checkpoint is bitwise identical to uninterrupted deterministic continuation under the same runtime.

## Multicell synapses and plasticity

`SynapseNetworkPlan` fixes cell, compartment, relation, and delay capacities. Initial `SynapseConnection` objects use either `CurrentSynapse` or `ConductanceSynapse`. Prepared state holds active masks, endpoints, parameters, exponentially decaying activation, and a fixed delay ring. A transition returns per-cell/per-compartment conductance and affine current offset arrays for cable inputs.

Dynamic synaptogenesis is explicit:

1. create a `SynapseRelationEvent` (`ACTIVATE` or `DEACTIVATE`);
2. call `evaluate_synapse_relation_event` to produce a candidate and capacity/endpoint/parameter status;
3. call `commit_synapse_relation_event` to atomically accept valid candidates or preserve the prior state.

A slot value of `-1` deterministically allocates the lowest inactive slot. Full capacity reports `CAPACITY_EXCEEDED`; compiled state shape never changes.

`PairSTDPPlan` implements bounded nearest-pair plasticity. Pre/post traces decay exactly, each corresponding spike replaces rather than accumulates its trace, and every trace is bounded by `trace_bound`. Active relation weights remain within explicit minimum/maximum bounds. `evaluate_pair_stdp` and `commit_pair_stdp` form an atomic candidate/evidence/commit transition so rejected plasticity never partially changes weights or traces. When relation events and STDP are coupled, `commit_synapse_relation_event_with_plasticity` atomically commits the relation and clears the reused slot's traces.

## Ion concentrations and Nernst dynamics

`IonDynamicsPlan` fixes ionic species, integer valences, intracellular/extracellular compartment volumes, temperature, minimum concentration, and conservation tolerances. `nernst_potential_mV` evaluates

```text
E = 1000 R T / (z F) * log(c_out / c_in)
```

An outward ionic current removes intracellular moles and adds exactly the same moles to the extracellular volume. The transition requires a scalar positive `dt_ms` and promotes integer current/concentration inputs to an inexact concentration dtype before applying a fractional step. Candidate evidence reports total moles before/after, per-species/compartment mole residual, intracellular electrical-charge residual, minimum concentration, and status. Nonpositive or nonfinite candidates fail closed. `sodium_potassium_pump_ion_currents` routes one net pump current into 3 Na⁺ outward and 2 K⁺ inward components.

## Stochastic channel populations

`MarkovChannelPlan` validates a continuous-time generator: finite square matrix, nonnegative off-diagonals, nonpositive diagonal, and zero row sums. Preparation exponentiates it for one fixed time step. Initial per-state counts and per-compartment totals must fit signed `int32` storage, so exact populations can never wrap during narrowing or summation. `evaluate_stochastic_channel_transition` draws exact integer multinomial source-state transitions using deterministic split/fold-in keys. It returns population-conservation evidence and `PRNGLineage(parent_key, draw_key, next_key, draw_index)`. The next key is consumed only when `commit_stochastic_channel_transition` accepts the candidate. Equal checkpoint state and key therefore reproduce equal counts and lineage.

## SWC adapter

`parse_swc_text` and `parse_swc_file` are host-only. They require seven fields per record, finite coordinates, positive radii and node IDs, one root, existing parents, an acyclic connected tree, and nonzero segment lengths. They construct stable `swc-{node_id}` compartment IDs and maximal branch paths. `SWCAdaptation.report` is the canonical `interchange.AdapterReport`: it declares the dropped absolute embedding and node-type assignment, synthesized root length, radius-to-diameter transform, source/target identities, coordinate mapping, preserved fields, and unit assumptions. Generic `require_lossless` consumers therefore reject the declared-loss import. `SWCAdaptation.evidence` provides the stable mapping, node/segment/branch counts, total segment length, node types, warnings, morphology identity, and content-sensitive evidence identity.

## Benchmark

The benchmark prepares a branched morphology, ordered leak/HH/pump program, a fixed-capacity ring synapse network, and multiple cell states. It separately reports lowering, compilation, synchronized execution, compiler cost/memory evidence, logical bytes, solve residual, and success:

```console
python benchmarks/electrophysiology.py --cell-counts 4 16 --warmup 2 --repeats 10
```

Use `--output path.json` to write a report. The repository does not store generated benchmark results.
