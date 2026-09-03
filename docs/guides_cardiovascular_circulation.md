# Cardiovascular circulation

PhydraX circulation models are acausal pressure/flow networks built directly on the generic dynamics DAE substrate. The circulation package declares components and connections; `AcausalDAESource`, structural analysis, index handling, and residual compilation remain owned by `phydrax.dynamics`. There is no circulation-specific DAE compiler. Likewise, zero-dimensional R/C/L and RCR Windkessel elements live only in circulation; hemodynamics couples to their ports instead of defining another terminal model.

## Kernel conventions

Circulation quantities use the cardiovascular kernel scale:

| Quantity | Kernel unit |
|---|---|
| time | ms |
| pressure | kPa |
| volume flow | mm³/ms |
| volume | mm³ |
| resistance | kPa·ms/mm³ |
| compliance | mm³/kPa |
| inertance | kPa·ms²/mm³ |
| pressure-volume work | kPa·mm³ |

A two-port component has ports named `inlet` and `outlet`. Each generic `DAEPort` exposes one pressure potential and one directed flow. Positive flow is from inlet to outlet. Fully qualified port IDs are `<component>.inlet` and `<component>.outlet`; `PressureFlowConnection` lowers their equality and flow orientation to `DAEConnection(..., (1, -1))`. Pressure sources prescribe a positive pressure rise from inlet to outlet. Compliances and elastance chambers state their unstressed volume and pressure reference explicitly.

## Components and networks

The component set includes:

- `Resistance` (`R`): Δp = Rq.
- `Compliance` (`C`): V - V₀ = C(p - p_ref) with dV/dt = q_in - q_out.
- `Inertance` (`L`): Δp = L dq/dt.
- `WindkesselRCR` (`RCR`): proximal resistance, circulation-owned compliant storage, and distal resistance.
- `PressureSource` and `FlowSource`: constant or identified callable waveforms.
- `TimeVaryingElastance`: a volume-storing chamber with p - p_ref = E(t)(V - V₀).
- `MechanicsChamberCoupling`: a storage-free pressure/flow adapter driven by a mechanics-owned volume rate.

Every component is a `PressureFlowComponent` wrapping a canonical `DAEComponent`. Its stable `component_id` includes model parameters, ownership, port names, and waveform identifiers. Custom waveform callables require a caller-supplied stable identifier.

```python
from phydrax.applications.cardiovascular import circulation as cv

pump = cv.PressureSource("pump", 10.0)
load = cv.Resistance("load", 2.0)
network = cv.CirculationNetwork(
    (pump, load),
    (
        cv.PressureFlowConnection("pump", "outlet", "load", "inlet"),
        cv.PressureFlowConnection("load", "outlet", "pump", "inlet"),
    ),
)
```

A network rejects unknown endpoints, multiply connected ports, duplicate component names, or ports without exactly one pressure and one flow coordinate. `network.closed` is true only when every port occurs exactly once. `network.source` is the `AcausalDAESource` consumed by generic dynamics tooling.

## Consistent initialization

Initialization follows plan → prepare → fixed-shape solve → evidence:

```python
plan = cv.ConsistentInitializationPlan(
    maximum_differentiations=2,
    maximum_tears=32,
)
prepared = cv.prepare_consistent_initialization(network, plan)
result = cv.initialize_consistent_state(prepared)
```

Preparation performs generic structural analysis and compilation and fails closed on a nonsquare or structurally singular network. Execution holds differential storage states fixed while solving algebraic states and consistent differential rates together through the native nonlinear root substrate. `result.evidence` retains the original declared DAE residual, physical and scaled residual norms, finiteness, nonlinear status, and the success decision. `circulation_state_values(result)` returns named zeroth-order values without replacing the retained reconstruction jet.

For the source/load example the consistent flow is 5 mm³/ms. The additive pressure gauge is determined by the algebraic solve, while the 10 kPa source rise and 2 kPa·ms/mm³ load law are satisfied exactly.

## Valve routes

Valve fidelity is represented by distinct types rather than string modes:

- `SmoothValve` blends open and closed resistance with a differentiable sigmoid. Use it for fixed-topology gradients and optimization.
- `ComplementarityValve` applies a scaled Fischer–Burmeister condition to nonnegative forward flow and valve slack. Zero smoothing is the exact complementarity route; positive smoothing regularizes the corner.
- `EventValve` uses opening and closing thresholds plus a minimum dwell time. `propose_event` returns an immutable candidate carrying the exact source-state identity. `commit_event` accepts only a candidate from the valve's current state, retains the caller's current state on rejection, and constructs the accepted coefficient state without changing DAE incidence or topology.

Discrete event decisions are a deliberate differentiation boundary. Do not differentiate through event selection. Differentiate each fixed-topology interval, retain `ValveEventCandidate` evidence, commit on the host, and record accepted transitions in `ValveEventLedger`. The ledger checks finite chronological events, per-valve alternation, and minimum dwell time.

## Reference closed loops

`systemic_closed_loop`, `pulmonary_closed_loop`, and `biventricular_closed_loop` assemble volume-conserving chamber/valve/vascular rings. `coronary_closed_loop` adds epicardial and microvascular resistance, an intramyocardial compliance, venous resistance, and smooth phasic extravascular pressure. These builders are transparent reference parameterizations in kernel units, not patient-specific defaults. Replace parameters during case preparation and preserve the declared units and reference pressures.

Each reference model retains a stable ID, its canonical network, chamber names, circulation-owned storage IDs, and a reference total volume. The coronary compliance uses transmural pressure p_vascular - p_extravascular(t), so systolic compression changes volume and flow without changing network topology.

## Mechanics-owned chamber replacement

A chamber volume has exactly one owner. Native elastance and compliance chambers declare `StorageOwner.CIRCULATION` and include a DAE volume variable. A `MechanicsChamberCoupling` declares `StorageOwner.MECHANICS`, contains no circulation volume variable, and contributes the mechanics volume rate to q_in - q_out.

```python
coupling = cv.MechanicsChamberCoupling(
    "left_ventricle",
    mechanics_chamber.chamber_id,
    mechanics_volume_rate,
)
model = cv.replace_chamber_with_mechanics(
    cv.systemic_closed_loop(),
    "left_ventricle",
    coupling,
    mechanics_initial_volume,
)
```

Replacement preserves the component name and inlet/outlet IDs, removes the circulation volume state, creates one external storage record, and updates the reference total volume. It rejects replacement of a non-circulation owner, a renamed component, or an adapter that declares duplicate storage.

## Periodic closure and pressure-volume work

Forced cardiac cycles use fixed-period shooting. `PeriodicShootingPlan` fixes cycle length and state shape; `prepare_periodic_shooting` binds an identified differentiable one-cycle map; `solve_periodic_shooting` solves Φ_T(x) - x = 0 with the native nonlinear solver. The returned `PeriodicShootingCandidate` contains terminal state, closure residual, relative closure, nonlinear status, and success evidence. `commit_periodic_state` fails closed unless that evidence is finite and successful.

`pressure_volume_work` evaluates -∮p dV, positive for chamber work delivered to blood. Supply a closed, ordered cycle including its repeated endpoint. `audit_pressure_volume_cycle` also reports pressure and volume closure and stroke volume.

## Conservation and passivity evidence

Use the ledgers at retained time samples:

- `audit_total_volume` checks V(t) - V(0) = ∫(Q_in - Q_out)dt. Omit boundary flow for a closed loop.
- `audit_passivity` checks ΔE + D - W_in ≤ tolerance using integrated input and nonnegative dissipated power.
- `audit_valve_events` checks deterministic committed event history.
- `audit_pressure_volume_cycle` checks cycle closure and chamber work.

All ledgers retain arrays and tolerances, expose explicit finite/success booleans, and fail closed on malformed axes or nonfinite input. Numerical tolerances are relative to documented ledger scales; they are not author assertions of conservation.

Run the qualification campaign from the repository root with `python tools/cardiovascular_circulation_qualification.py`. The benchmark `benchmarks/cardiovascular_circulation.py` records closed-loop construction and structural-analysis time, consistent initialization, and compiled batched smooth-valve throughput.
