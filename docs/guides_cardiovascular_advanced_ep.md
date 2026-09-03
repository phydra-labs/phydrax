# Advanced cardiovascular electrophysiology

PhydraX provides three deliberately separate propagation layers:

1. the anisotropic eikonal solver computes reduced earliest-arrival fields;
2. the fixed-capacity Purkinje runtime resolves discrete network events; and
3. the bidomain FEM advances transmembrane voltage and extracellular fields.

An eikonal arrival field or Purkinje event history is not a tissue action-potential
state. Use them to initialize, trigger, or constrain a reaction--diffusion model;
do not substitute either reduced route for ionic tissue electrophysiology.

## Units, signs, and supports

The cardiovascular kernel uses millimetres and milliseconds. Eikonal squared-
velocity tensors therefore have units `mm2/ms2`. Conductivity tensors use
`mS/mm`; voltage and potential use `mV`; nodal reaction and pacing data use
`uA/mm3`. PMJ coupling conductance uses `mS`, so `mS * mV = uA` without a hidden
conversion.

`BidomainStepInputs.ionic_current_uA_per_mm3` is outward-positive. The
transmembrane pacing/stimulus field is inward-positive and is subtracted from
the outward ionic current in the capacitive balance. PMJ pair current is positive
from Purkinje support to tissue support. Every support is an explicit stable-ID
or fixed node-index array; inactive fixed-capacity slots use stable ID `-1`.

## Anisotropic arrival times

`GraphEikonalRoute` and `FiniteElementEikonalRoute` are distinct route types.
The graph route preserves the supplied undirected edges. The FEM route derives a
complete fallback edge stencil and causal affine-simplex roots on each line,
triangle, or tetrahedron while retaining the simplex topology identity. A local
root solves `grad(T)^T C grad(T) = 1`; a noncausal root is rejected in favour of
the anisotropic edge update, as required on obtuse characteristics.

For edge displacement `dx` and the endpoint-averaged squared-velocity tensor
`C`, preparation computes the travel delay

```text
dt = sqrt(dx^T C^-1 dx).
```

Thus, a fibre-axis eigenvalue of `4 mm2/ms2` corresponds to speed `2 mm/ms`.
The solver runs a fixed number of synchronous graph/simplex sweeps. It reports
reachability, the maximum update defect, deterministic diagnostic predecessors,
and the smallest selected-update margin.

```python
import jax.numpy as jnp
from phydrax.applications.cardiovascular.electrophysiology import (
    AnisotropicEikonalPlan,
    GraphEikonalRoute,
    solve_anisotropic_eikonal,
)

route = GraphEikonalRoute(
    jnp.asarray((10, 20, 30)),                 # stable node IDs
    jnp.asarray((100, 101)),                   # stable edge IDs
    jnp.asarray(((0, 1), (1, 2))),
)
prepared = AnisotropicEikonalPlan(
    route,
    jnp.asarray(((0.0, 0.0), (2.0, 0.0), (5.0, 0.0))),
    jnp.asarray(((4.0, 0.0), (0.0, 1.0))),
).prepare()
result = solve_anisotropic_eikonal(prepared, jnp.asarray((0,)), jnp.asarray((0.0,)))
# result.arrival_time_ms == [0.0, 1.0, 2.5]
```

Topology preparation is a derivative boundary. Within one prepared topology,
arrival derivatives are qualified only when
`evidence.fixed_topology_derivative_valid` is true. A shortest-path tie clears
that flag even though stable IDs still make the primal predecessor deterministic.

## Purkinje events

`PurkinjeNetworkPlan` owns fixed node, edge, stimulus, and output capacities.
Active edge delays are physical delays in milliseconds. `PurkinjeNetworkState`
stores latest activation, refractory deadlines, activation counts, and a dynamic
block mask. `propagate_purkinje` is a host-side proposal/evidence/commit
transaction:

- priority is `(time, stable node ID, stable event ID, insertion order)`;
- an arrival before the target refractory deadline is recorded and rejected;
- a blocked edge records an `EDGE_BLOCK` outcome and launches no wave;
- antiparallel waves with overlapping edge transit collide at their analytic
  meeting time and both node-arrival events are cancelled;
- overflow leaves the accepted state unchanged and exposes the candidate state.

```python
from phydrax.applications.cardiovascular.electrophysiology import (
    initialize_purkinje_state,
    make_purkinje_stimulus_batch,
    propagate_purkinje,
    PurkinjeNetworkPlan,
)

network = PurkinjeNetworkPlan(
    jnp.asarray((10, 20, 30)),
    jnp.asarray((100, 101)),
    jnp.asarray(((0, 1), (1, 2))),
    jnp.asarray((1.0, 2.0)),
    250.0,
    event_capacity=64,
    stimulus_capacity=4,
)
state = initialize_purkinje_state(network)
stimuli = make_purkinje_stimulus_batch(network, (1,), (0,), (0.0,))
transaction = propagate_purkinje(network, state, stimuli)
```

Event ordering, refractory switches, block switches, collisions, and overflow
are discrete derivative boundaries. The runtime never differentiates through
host event mutation. `fixed_event_sequence_derivative_valid` is intentionally
conservative: it requires a successful, strictly separated sequence without a
realized branch event.

## PMJ exchange and delayed activation

`PMJExchangePlan` binds stable junction IDs to Purkinje and tissue node support
and records the exact `PurkinjeNetworkPlan.plan_id`. `evaluate_pmj_exchange`
computes one ohmic current per junction and scatters exact equal-and-opposite
nodal currents. Conservation evidence includes the net exchange current and
pair balance error.

`schedule_pmj_activations` consumes a successful `PurkinjePropagationResult`,
never a bare event batch. It rejects failed transactions and results from any
network other than the PMJ-bound plan before adding junction delays. Resulting
tissue activations are sorted deterministically and checked against tissue
refractory deadlines. Rejected PMJ arrivals remain in the fixed output batch
with `accepted=False`; they are not silently discarded. Capacity or causality
failures set status and fail closed.

## Open-loop and demand pacing

A `PacingProtocol` combines fixed pulse arrays with exactly one route type:
`TissuePacingTarget` produces a nodal current-density field, while
`PurkinjePacingTarget` produces onset events through
`pacing_purkinje_stimuli`. Route choice is a type, never a string mode.

`DemandPacingControllerPlan` defines an escape interval, bounded feedback cycle
length, pulse support, duration, and amplitude. Its immutable controller state
records the last sensed activation, next deadline, current cycle length, command
count, and last update. `step_demand_pacing_controller` processes a valid sensed
activation before deciding whether one escape pulse is due. A nonmonotone update
or invalid sensing time retains the prior accepted state.

## Gauge-fixed bidomain and torso coupling

`BidomainFEMPlan` assembles affine-P1 mass and anisotropic stiffness operators on
line, triangle, or tetrahedral heart meshes. Use `HeartOnlyBidomainRoute` for a
heart-only solve. `HeartTorsoBidomainRoute` adds a torso volume-conductor mesh and
fixed interface pairs with physical coupling conductances.

For heart mass matrix `M`, intracellular stiffness `Ki`, extracellular stiffness
`Ke`, capacitance density `C`, and time step `dt`, the heart-only implicit system
is

```text
(C/dt M + Ki) Vm + Ki phi_e = C/dt M Vm_old + M(I_stim - I_ion)
Ki Vm + (Ki + Ke) phi_e     = M I_e.
```

The heart--torso route augments the extracellular block with torso stiffness and
pair contributions `g * (phi_e - phi_t)`. Those terms enter heart and torso with
opposite signs. Interface evidence reports potential-jump norm, interface-current
norm, and the equal-and-opposite flux balance error.

Pure-Neumann extracellular operators have a constant nullspace. PhydraX does not
pin an arbitrary node. It appends one weighted-mean Lagrange multiplier and
solves the resulting DAE saddle system. `BidomainGaugeEvidence` reports the
ungauged constant-mode defect and the accepted weighted-mean constraint. A net
incompatible extracellular/torso source fails closed rather than being hidden by
the multiplier.

```python
from phydrax.applications.cardiovascular.electrophysiology import (
    BidomainFEMPlan,
    HeartOnlyBidomainRoute,
    initialize_bidomain_state,
    step_bidomain,
    zero_bidomain_inputs,
)

prepared = BidomainFEMPlan(
    HeartOnlyBidomainRoute(),
    jnp.asarray((10, 20, 30)),
    jnp.asarray((100, 101)),
    jnp.asarray(((0.0,), (1.0,), (2.0,))),
    jnp.asarray(((0, 1), (1, 2))),
    jnp.asarray(((1.0,),)),                 # intracellular mS/mm
    jnp.asarray(((2.0,),)),                 # extracellular mS/mm
    dt_ms=0.1,
    membrane_capacitance_uF_per_mm3=1.0,
).prepare()
state = initialize_bidomain_state(prepared, jnp.asarray((0.0, 1.0, 0.0)))
step = step_bidomain(prepared, state, zero_bidomain_inputs(prepared))
```

The solve returns norms for every field block, the gauge row, interface transfer,
and the total relative residual. The reusable block preconditioner solves the Vm
block and a rank-one gauge-stabilized extracellular/torso block through the
PhydraX linear algebra runtime. Its evidence records diagonal scaling and the
actual defect of the preconditioned action.

When `sigma_e = r sigma_i`, `step_proportional_monodomain_limit` independently
solves the analytic reduction with effective conductivity
`r/(1+r) sigma_i`. It rejects torso routes and extracellular forcing. Comparing
its voltage against the full gauge-fixed bidomain step is the supported
monodomain-limit qualification, not a string-selected bidomain approximation.

## Qualification and performance evidence

The advanced qualification covers analytic anisotropic travel time,
antiparallel Purkinje collision timing, PMJ conservation/support, gauge
nullspace removal, heart--torso flux balance, and the proportional-conductivity
monodomain limit:

```text
python tools/cardiovascular_advanced_ep_qualification.py
```

The benchmark separately compiles and measures full gauge-fixed bidomain and
analytic monodomain-limit steps on the same fixed one-dimensional FEM problem:

```text
python benchmarks/cardiovascular_bidomain.py --nodes 128 --warmup 2 --repeats 10
```

Benchmark output includes environment identity, compiler cost/memory evidence,
per-run synchronized timing samples, logical array sizes, and solve residuals.
