# Electrophysiology

The electrophysiology application provides native compartmental cable cells, physical LIF/AdEx neurons, scheduled spike sources, synaptic transport and learning, ion concentrations, and stochastic channel populations. Host plans validate identities, topology, and capacities; prepared runtimes carry fixed-shape JAX state. A network can mix cell types and compartment counts without padding every cell to the largest morphology.

This is a physical neural runtime, not a surrogate-gradient artificial spiking layer. Regional neural-mass models and hemodynamic/BOLD observation belong to the separate `phydrax.applications.neuroscience` package. Membrane voltage is not a BOLD signal.

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

`ELECTROPHYSIOLOGY_UNITS` stores these eight canonical `UnitDefinition` objects, not unit strings. `convert_quantity(value, from_unit, to_unit)` accepts either those objects or a token from the application's closed alias table and rejects different dimensions, different reference systems, and unknown offset or logarithmic units rather than inferring a conversion. Conversion multipliers used by morphology, mechanisms, and ion dynamics are derived once while constructing or preparing immutable plans; compiled kernels still receive only raw homogeneous arrays. Unit IDs are bound into physical plan IDs.

Membrane and synaptic current is **outward positive**. Injected current and current-clamp amplitude are **inward positive**. Thus an affine membrane current is

```text
I_out(V) = conductance_uS * V_mV + current_offset_nA
```

and a conductance synapse has `current_offset_nA = -conductance_uS * reversal_mV`. Exact voltage clamps report the inward clamp current required by the physical capacitance, axial, membrane, and stimulus balance.

A positive `CurrentSynapse.current_scale_nA` is outward and therefore hyperpolarizing in a passive cell; a negative scale is inward. Synaptic weights are nonnegative, so current-synapse polarity belongs in the scale. A conductance synapse's effect depends on `V_mV - reversal_mV`, not an excitatory/inhibitory label alone.

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

Preparation computes cylindrical membrane areas, capacitances, interface axial conductances, axial diagonal coefficients, and a reusable native `phydrax.linalg.TreeTopology`. Morphology storage is linear in compartment count: a parent-edge representation replaces a stored dense Kirchhoff matrix. The tree must remain fixed during execution; changing compartment topology requires a new plan.

## Membrane programs and cable solves

`MembraneMechanism` defines steady-state gate initialization, gate updates, and outward-current affine coefficients. `MembraneProgram` evaluates mechanisms in deterministic order. `MembraneProgramState.gates` is a **tuple**, with entry `i` shaped `[compartment_count, program.mechanisms[i].gate_count]`. Zero-gate mechanisms have a zero-width array; custom mechanisms are not restricted to a common padded gate count. Built-in mechanisms are:

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

Backward Euler and Crank–Nicolson assemble the theta-method system for membrane coefficients frozen at the start of the segment. `assemble_cable_system` returns native `TreeLinearOperator` objects and vectors; `step_cable` uses `phydrax.linalg` structured direct solving without materializing a dense cable matrix. Exact Dirichlet row replacement affects the solve operator, while the unchanged physical operator supplies Kirchhoff and clamp-current evidence.

Each result reports absolute/relative linear residual, per-compartment Kirchhoff residual, global charge-balance residual, inward clamp current, finiteness, nonlinear routing, and a bitwise status. Voltage, gates, and time commit together only on success. Built-in HH gate updates use the affine exponential solution at the newly computed voltage, not forward Euler; this split update is not an exact solution of the fully coupled nonlinear HH system. Nonfinite updated gates reject the whole step. `step_cable(..., elapsed_ms=h)` uses one positive physical interval consistently for voltage, gates, charge balance, and time; omission uses the prepared cable interval.

The mechanism contract requires an affine voltage current for the frozen state. It does not turn arbitrary voltage-nonlinear mechanisms into an implicit nonlinear solve. The pump's concentration dependence is evaluated from the supplied concentrations, and ion routing must be bound explicitly as described below.

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

Single-cell checkpoints are host-created and content-addressed over the complete cable and recording state plus prepared protocol identity. Restore rejects provenance or content mismatches. Use the same prepared runtime and numerical environment for deterministic replay.

## Heterogeneous neural networks

`NeuralCellPlan(cell_id, model, ...)` accepts a prepared cable solver, `LeakyIntegrateAndFire`, `AdaptiveExponentialIntegrateAndFire`, or `SpikeSource`. Cell IDs are stable; point cells expose one `"soma"` compartment and sources one `"source"` endpoint. Compatible physical models are grouped for vectorized execution, but endpoint arrays remain flat in plan cell order and then compartment order. `SynapseNetworkPlan.offsets` maps each cell into that vector.

Each cell has one observable spike site. Cable detectors, including detectors on HH cells, **observe upward threshold crossings without resetting voltage or gates**. `detector_compartment`, `threshold_mV`, and `rearm_mV` select the site and hysteresis. A detector disarms after firing and rearms only below threshold and at/below the rearming level; a plateau does not repeatedly fire.

LIF/AdEx thresholds belong to the point model and trigger an explicit voltage reset and absolute refractory deadline. AdEx additionally increments adaptation; adaptation continues during the refractory voltage hold. AdEx's `exponential_threshold_mV` is the exponential-current onset, distinct from the upper spike cutoff `threshold_mV`. LIF integrates constant-input charge analytically, including the zero-conductance limit. AdEx uses a numerical exponential-midpoint segment map; values continued above its cutoff serve root bracketing, not a physical post-spike waveform.

`SpikeSource` emits only its scheduled `external_spikes=(time_ms, cell_id)` events. It is not a physical membrane and cannot be driven to spike by injected current. `neural_voltage` and network recordings include a numeric source placeholder for fixed shape; always use `runtime.physical_voltage_mask` to exclude it from voltage analyses.

### A source driving a physical neuron

This public-API example schedules off-grid emissions and a physical delay in event mode:

```python
import jax.numpy as jnp
from phydrax.applications import electrophysiology as ep

cells = (
    ep.NeuralCellPlan("input", ep.SpikeSource()),
    ep.NeuralCellPlan(
        "target",
        ep.LeakyIntegrateAndFire(0.2, 0.01, -65.0, -50.0, -65.0, 2.0),
    ),
)
synapses = ep.SynapseNetworkPlan(
    (1, 1),  # Compartment count of each cell, not a rectangular cell capacity.
    4,       # Fixed relation capacity, including unused slots.
    5.0,     # Maximum physical delay in ms.
    0.1,     # Requested network step in ms.
    execution="event",
    connections=(
        ep.SynapseConnection(
            "input-to-target", 0, 0, 1, 0,
            ep.CurrentSynapse(5.0, -1.0),
            delay_ms=0.35,
            weight=1.0,
        ),
    ),
)
runtime = ep.NeuralNetworkPlan(
    cells, synapses,
    external_spikes=((1.03, "input"), (6.03, "input")),
    queue_capacity=64,
    spike_capacity=64,
    recording_capacity=256,
    maximum_events_per_step=128,
).prepare()
state = ep.initialize_neural_network(runtime)
first = ep.step_neural_network(runtime, state)
if not bool(first.evidence.successful):
    raise RuntimeError(f"Neural step rejected: {first.evidence.status}")

run = ep.run_neural_network(runtime, first.state, 199)
if bool(jnp.any(run.status != 0)):
    raise RuntimeError("Neural continuation rejected a step")

checkpoint = ep.checkpoint_neural_network(runtime, run.state)
restored = ep.restore_neural_network(runtime, checkpoint)
continued = ep.run_neural_network(runtime, restored, 20)
physical_voltage = ep.neural_voltage(runtime, continued.state)[
    runtime.physical_voltage_mask
]
```

The sample checks primal acceptance, not derivative validity. For a mixed cable network, pass a prepared cable solver as another cell's model and put its actual compartment count in `compartment_counts`. An explicit initial `voltage_mV` must be a finite vector of length `endpoint_count`; point-neuron voltages must start below their cutoff.

`zero_neural_inputs(runtime)` constructs neutral flat inputs. Supply `NeuralNetworkInputs` to `step_neural_network` for inward current, cable voltage-clamp masks/targets, and modulation. `run_neural_network(..., inputs=inputs)` holds the same inputs across its requested steps; for time-varying inputs, call the step function with the appropriate input at each boundary. Scheduled plan clamps are `(cell_id, CurrentClamp(...))` or `(cell_id, VoltageClamp(...))` pairs. Exact voltage clamps are cable-only, not point-neuron resets.

### Event versus clock execution

`SynapseNetworkPlan(..., execution="event")` selects bounded physical-time event execution. Within each requested `dt_ms`, the runtime stops at pending deliveries, external emissions, scheduled clamp boundaries, channel draw clocks, refractory releases, and bounded root-search subdivisions. It localizes a bracketed upward crossing on the numerical cell segment, advances to the selected time, and processes the boundary before continuing.

At an event boundary the order is: advance continuous state and decay activation; gather endogenous and external spikes; enqueue emissions with the current weights; deliver due messages, including zero-delay emissions; apply point resets and detector disarming/rearming; update learning traces; perform due channel draws. Emissions therefore precede same-boundary learning. Queue ordering is deterministic by time, relation slot, and insertion sequence. Continuous current/conductance synapses change activation at delivery, not membrane voltage instantaneously.

Root isolation is **bounded refinement, not certification of every hidden crossing**. `root_subdivisions` limits segment length, and `root_iterations` bounds refinement of observed brackets. A rise and fall entirely inside a segment can evade endpoint crossing detection. `root_tolerance_ms`, numerical precision, and the segment integrator constrain localization accuracy. Refine the time resolution and examine scientific convergence; a successful status is not proof of event completeness.

`execution="clock"` instead detects and resets at tick boundaries without sub-tick root localization. Physical connection delays and external emission times must be representable on the clock grid; off-grid values are rejected, not silently rounded. Use event mode for off-grid stimulation or independently timed auxiliary draws. Clock execution is a distinct discretization, not an event simulation with equivalent spike times.

Both modes enforce fixed queue, spike-recording, voltage-recording, and work capacities. `step_neural_network` either advances the requested interval and appends one voltage sample, or retains the entire previous state. This includes cells, gates, ions, channels and PRNG lineage, relations, learning traces, queue, detectors, and recordings. `NeuralNetworkEvidence` reports status, `successful`, `sensitivity_valid`, event batches, emitted spikes, delivered messages, and maximum queue occupancy. Capacity exhaustion, cell/root/learning/auxiliary failure, or bounded event work exhaustion are explicit failures; the engine does not nudge time or drop events to continue. `run_neural_network` retains the last committed state after its first failed step and returns per-step status arrays.

## Physical synapses and relation lifetimes

`SynapseNetworkPlan(compartment_counts, synapse_capacity, maximum_delay_ms, dt_ms, ...)` replaces rectangular cell/compartment capacity assumptions. `SynapseConnection` and `SynapseRelationEvent` take canonical physical `delay_ms`. Relations contain active masks, logical and flat endpoints, model parameters, weights, exponentially decaying activation, and separate version/generation counters.

`CurrentSynapse(tau_ms, current_scale_nA)` contributes an outward affine offset. `ConductanceSynapse(tau_ms, conductance_scale_uS, reversal_mV)` contributes `g` and `-g * reversal_mV`. Arrival activation is weighted at **emission**, not reweighted at delivery or on every subsequent decay. Learning cannot retroactively change an in-flight spike. The network uses the interval-average decaying activation for its frozen synaptic drive; this does not make a time-varying conductance/voltage interaction analytically exact.

The standalone synapse API separates `SynapseRelationState` from clock `SynapseTransportState`. `evaluate_synapse_network_transition` and `commit_synapse_network_transition` operate the standalone clock transport; event-mode neural networks compose relations with their bounded priority queue instead. `synapse_drive` returns flat affine coefficient vectors, not rectangular per-cell arrays.

For a neural network, apply structural changes only at a committed boundary with `apply_neural_relation_event(runtime, state, event)`. `ACTIVATE` with slot `-1` deterministically chooses the lowest inactive slot; full capacity rejects the transaction. `DEACTIVATE` cancels pending deliveries, clears activation and learning traces, and changes lifetime generation. Reusing a slot cannot deliver events from its previous lifetime. Weight updates change `relation_version`, not lifetime `generation`.

For standalone composition, `evaluate_synapse_relation_event` produces the candidate. Use a relation-only commit only when no transport needs updating. `commit_synapse_network_relation_event` also clears clock-ring deliveries; the corresponding `..._with_plasticity` helpers include trace/eligibility clearing. Mutating relation fields alone is not a substitute for a structural transaction.

## Pair STDP, eligibility, and modulation

`PairSTDPPlan` implements bounded nearest-pair traces with exact physical-time decay. Each event replaces, rather than adds to, its trace, capped by `trace_bound`. Simultaneous pre/post events pair with earlier traces, not each other. `pairing="emission"` uses presynaptic endpoint spikes; `"arrival"` uses **unweighted per-relation arrival counts**, allowing learning even at zero weight. `weight_dependence="additive"` or `"soft-bound"` selects the update law; soft bounds scale potentiation/depression by normalized distance to the corresponding bound raised to `weight_exponent`. Weights remain within the explicit minimum/maximum.

`EligibilitySTDPPlan(pair_stdp, eligibility_time_constant_ms, ...)` accumulates signed, bounded pair eligibility with exponential decay. A signed modulation impulse changes weight by `learning_rate * modulation * eligibility`; zero modulation preserves delayed credit without applying that update. Eligibility persists after reward or punishment until it decays or the relation is deleted.

`NeuralNetworkPlan.modulation_scope` is `"global"` (scalar), `"post"` (flat endpoint vector), or `"relation"` (synapse-capacity vector). In the neural runtime, `inputs.modulation` is applied **once at the end of the requested network step**, after its pair events and eligibility decay, not once per internal root subdivision. It is an impulse, not a continuously integrated rate. Holding a nonzero value across `run_neural_network` repeats that impulse on each accepted step.

Standalone `evaluate_pair_stdp` / `evaluate_eligibility_stdp` accept `elapsed_ms` and, for arrival pairing, `presynaptic_arrivals`; their commit functions accept weights and learning state atomically or preserve both. Endpoint event arrays have length `endpoint_count`, arrival arrays length `synapse_capacity`. Structural lifetime changes clear both pair traces and eligibility.

## Differentiation and scientific validity

Cable solves use the native structured linear-solve differentiation contract for forward JVPs and reverse VJPs; no dense cable or reset Jacobian is required. Event-time sensitivities differentiate the **numerical segment map** used to locate the crossing. For `F(t, p) = guard(t, state_at_time(t, p))`, the selected branch satisfies `dt/dp = -partial_p F / partial_t F`. This is not automatically the continuous-vector-field saltation derivative: substituting a physical vector field for the numerical time derivative changes the derivative being computed.

Differentiation is branchwise: hold morphology, mechanism order, capacities, relation occupancy, detector/reset choices, and event ordering fixed. It does not differentiate integer event counts, heap/slot selection, topology edits, or stochastic multinomial draws, and it supplies no score-function or surrogate-gradient estimator. Clock-mode gradients are those of the fixed-tick branch, not localized spike-time gradients.

For event execution, primal acceptance and sensitivity qualification are separate. Grazing roots, immediate threshold events, and coincident/nearby competing roots can invalidate sensitivities even when a primal step is accepted. Rejected or sensitivity-invalid event steps produce NaN tangents on inexact state, not an epsilon-regularized event slope. Check `evidence.sensitivity_valid` or the run's `sensitivity_valid` array before using derivatives. A true value qualifies the selected numerical branch; it does not certify unobserved crossings or robustness to an event-order change.

## Neural checkpoints and continuation

`checkpoint_neural_network` captures the complete committed state: grouped cells, mechanism gates, concentrations, stochastic counts and keys/draw clocks, relations and learning state, pending deliveries with emission amplitudes and lifetime generations, fanout, detector arming, external-source cursor, time, and recordings. `restore_neural_network` validates both the prepared runtime identity and state content fingerprint; it does not reconstruct a new runtime from a checkpoint.

Create/restore checkpoints on the host and continue with the same prepared runtime and numerical environment. A checkpoint does not resize remaining queue or recording capacity, erase past records, or store future call-supplied inputs. Supply the same future inputs for reproducible continuation.

## Ion concentrations and Nernst dynamics

`IonDynamicsPlan` fixes ionic species, integer valences, intracellular/extracellular compartment volumes, temperature, minimum concentration, and conservation tolerances. `nernst_potential_mV` evaluates

```text
E = conversion_factor(V, mV) R T / (z F) * log(c_out / c_in)
```

An outward ionic current removes intracellular moles and adds exactly the same moles to the extracellular volume. The transition requires a scalar positive `dt_ms` and promotes integer current/concentration inputs to an inexact concentration dtype before applying a fractional step. Candidate evidence reports total moles before/after, per-species/compartment mole residual, intracellular electrical-charge residual, minimum concentration, and status. Nonpositive or nonfinite candidates fail closed. `sodium_potassium_pump_ion_currents` routes one net pump current into 3 Na⁺ outward and 2 K⁺ inward components.

Network coupling is explicit: `NeuralIonCoupling(cell_id, prepared_ion_runtime, current, coupling_id=...)` binds a cable cell to a callback `current(old_cell, new_cell)` returning outward-positive `[species, compartment]` currents in nA. The required stable coupling ID binds that caller-owned scientific attribution into the network/checkpoint identity. Supply one intracellular and extracellular concentration array per binding to `initialize_neural_network`. Each segment updates concentrations and copies them back into that cell's cable state; failure rolls back the entire network step.

The application never infers ionic species from total membrane current. Choose species ordering, volumes, valences, and current attribution consistently with the actual membrane program; avoid routing pump or channel current twice. Concentrations affect a mechanism only through that mechanism's explicit use of its concentration inputs. Merely preparing ion dynamics or evaluating a Nernst potential does not replace a fixed reversal parameter in a leak, HH, synapse, or channel model.

## Stochastic channel populations

`MarkovChannelPlan` validates a continuous-time generator: finite square matrix, nonnegative off-diagonals, nonpositive diagonal, and zero row sums. Preparation exponentiates it for one fixed time step. Initial per-state counts and per-compartment totals must fit signed `int32` storage, so exact populations can never wrap during narrowing or summation. `evaluate_stochastic_channel_transition` draws exact integer multinomial source-state transitions using deterministic split/fold-in keys. It returns population-conservation evidence and `PRNGLineage(parent_key, draw_key, next_key, draw_index)`. The next key is consumed only when `commit_stochastic_channel_transition` accepts the candidate. Equal checkpoint state and key therefore reproduce equal counts and lineage.

`NeuralChannelCoupling(cell_id, prepared_channel_runtime, open_state, single_channel_conductance_uS, reversal_mV)` explicitly maps open counts to membrane conductance on that cell's compartments. Network initialization requires one `[compartment, channel_state]` count array per binding plus an explicit random key. Conductance is held between the prepared channel's draw times; event execution stops at those times rather than drawing again for every root-search evaluation. Counts, next draw time, and PRNG lineage participate in the enclosing atomic network commit.

This is a fixed-generator, fixed-draw-interval channel model. The runtime does not infer a voltage-dependent transition generator, automatically bind ionic species to a channel, or differentiate through discrete channel draws. Choosing a channel binding without an ion-current callback does not conserve species concentrations on its behalf.

## SWC adapter

`parse_swc_text` and `parse_swc_file` are host-only. They require seven fields per record, finite coordinates, positive radii and node IDs, one root, existing parents, an acyclic connected tree, and nonzero segment lengths. They construct stable `swc-{node_id}` compartment IDs and maximal branch paths. `SWCAdaptation.report` is the canonical `interchange.AdapterReport`: it declares the dropped absolute embedding and node-type assignment, synthesized root length, radius-to-diameter transform, source/target identities, coordinate mapping, preserved fields, and unit assumptions. Generic `require_lossless` consumers therefore reject the declared-loss import. `SWCAdaptation.evidence` provides the stable mapping, node/segment/branch counts, total segment length, node types, warnings, morphology identity, and content-sensitive evidence identity.

See the [SONATA guide](guides_sonata.md) for the supported host-only grouped nodes/edges/types/spikes import and semantic export. It maps supported native models to a neural plan with explicit stable endpoint identities and physical delays; it is not a foreign simulator/configuration runner. Cable section/position mapping must be supplied explicitly for supported prepared cable models.

## End-to-end example

The endogenous example stimulates a physical HH cable cell, detects its action
potential without resetting the cable, delivers that spike through a delayed
plastic synapse, checkpoints all coupled state, and continues the postsynaptic
point neuron:

```console
python examples/endogenous_neural_network.py
```

## Benchmark

The cable benchmark prepares a branched morphology and ordered leak/HH membrane
program, then reports lowering, compilation, synchronized execution,
unit/plan identities, voltage bounds, and solver evidence:

```console
python benchmarks/electrophysiology.py --steps 128 --warmup 2 --repeats 10
```

The network benchmark reports clock/event compilation, synchronized execution,
emissions, deliveries, queue occupancy, state storage, and accepted physics:

```console
python benchmarks/neural_network.py --cells 16 --steps 16 --modes clock event
```

Use `--output path.json` to write a report. The repository does not store generated benchmark results.
