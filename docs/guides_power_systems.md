# Balanced power systems

`phydrax.applications.power` composes balanced positive-sequence RMS networks with
native nonlinear, sparse-derivative, convex/structured optimization and index-one
DAE solvers. It is not an EMT solver, an unbalanced distribution solver, a complete
PSS/E implementation, or a full CGMES implementation. Unsupported import semantics
are rejected rather than translated into placeholder equipment or profiles.

## Bases, signs and electrical laws

All network powers and impedances are per unit. `PowerNetwork.base_mva` is **total
three-phase** MVA. `Bus.base_kv` is line-line RMS kV. Consequently:

- Zbase [ohm] = base_kv² / base_mva.
- Ibase [A] = 1000 × base_mva / (sqrt(3) × base_kv).
- Phase RMS Vbase [V] = 1000 × base_kv / sqrt(3).
- Sbase = 3 × phase Vbase × Ibase. Do not multiply a per-unit power by three again.

`PowerBase` supplies these conversions and explicit old/new-base impedance
rebasing. Machine impedances and H belong to the declared machine MVA base;
`base_mva=None` on a machine selects the network base explicitly.

The phasor convention is exp(+i wt). A component's terminal current points inward.
A bus's specified complex injection is generation minus load. `Generator.p/q` are
generation-positive; `Load.p/q` are demand-positive. Angles and phase shifts are
radians. Complex power into a passive branch is V × conj(I). A positive shunt g
consumes active power, and positive shunt b supplies reactive power.

A branch is a pi line or two-winding transformer with series y = 1/(r+i x), total
charging b, and from-side tap t = tap × exp(i phase), relative to the two bus voltage
bases. The four terminal coefficients, in `branch_admittance` column order, are:

- Yff = (y+i b/2) / |t|²
- Yft = −y / conj(t)
- Ytf = −y / t
- Ytt = y+i b/2

Both terminal currents are reconstructed independently. This matters for charging,
losses, nonunity taps and phase shifts: the to-end power is not generally the
negative of the from-end power. `Branch.rate` is the continuous **either-end** MVA
limit on network base; infinity means no declared rating. Offline branches retain
their result slot and contribute zero current.

These equations agree with the published [MATPOWER branch-admittance
model](https://matpower.org/docs/ref/matpower7.1/lib/makeYbus.html). The implementation
is native: `compile_network` builds a `SparseCoordinateOperator` from an
`EdgeRelation`. `CompiledNetwork.ybus` is an explicitly requested dense
materialization; normal power-flow residuals use its sparse current action.

Physical equipment and study controls are separate. `Bus(id, base_kv, v_min, v_max)`
contains no PQ/PV/reference label, setpoint or initial angle. A reusable
`PowerNetwork` is paired with `PowerStudy(controls)`, containing exactly one
`BusControl(bus, kind, voltage, angle)` per bus. `BusControl.voltage` is the PV /
reference setpoint and initial magnitude; `angle` is the initial/reference angle.
The controls may be listed in any order, but no missing controls are synthesized.
`Generator` does not carry a second competing voltage setpoint.

`compile_network(network, study)` binds those explicit controls once in a
`CompiledNetwork`, retaining both `.network` and `.study`. Solvers accept either
the physical network with `study=...` or that compiled binding. Supplying a new
study to an already-bound compilation is an error; bind the unchanged physical
network to the new study instead. Dynamic constitutive laws use the physical
network and explicit machine/infinite-source configuration, not permanent bus types.

Every connected island must have exactly one explicit reference `BusControl`.
Compilation rejects missing/duplicate references, incomplete study coverage,
zero-impedance branches and unknown IDs.
It does not choose slack buses or silently merge nodes. A generator-free reference
is an explicit ideal voltage boundary for power flow, whose power is recorded in
`external_reference_power`. OPF never adds an unpriced external source: each
island must have an explicit online generator and all bus balance equations hold.

## Rectangular AC power flow

```python
from phydrax.applications import power

network = power.PowerNetwork(
    (power.Bus("source", 110), power.Bus("load", 110)),
    (power.Branch("line", "source", "load", 0.0, 0.1),),
    (power.Generator("g", "source"),),
    (power.Load("d", "load", 0.5),),
    base_mva=100.0,
)
study = power.PowerStudy((power.BusControl("source", "reference"), power.BusControl("load")))
compiled = power.compile_network(network, study)
result = power.solve_power_flow(compiled)
print(result.status, result.residual_norm, result.total_balance)
```

For this lossless two-bus example, the high-voltage solution is
Vload = (1 + sqrt(1 − 4 × 0.05²))/2 − i 0.05. The from-end active flow is +0.5 pu,
the to-end active flow is −0.5 pu, and the series reactive loss is positive.

The native root uses real and imaginary voltage coordinates. PQ buses impose P/Q
balance, PV buses impose P and squared voltage magnitude, and references impose
the prescribed complex voltage. `solve_power_flow` places a bounded host active-set
controller around `nonlinear.implicit_root_result`; it does not implement a second
Newton method. Generator Q is allocated by bounded equal incremental participation
around each generator's requested Q. A PV bus exceeding its aggregate Q capability
becomes `q_min` or `q_max`; a bound mode is released when its voltage complementarity
sign reverses. `mode_history`, `modes` and `switching_buses` expose decisions. Repeated
mode sets, exhausted mode work and native nonlinear failures are explicit failures.

Reference P/Q demand is shared by the same bounded allocation rule. Reference limit
violations produce `reference_limit_failure`, preserving the reference rather than
turning it into PQ or inventing a different slack. Out-of-range fixed generator
dispatch also fails. Branch and voltage operating violations are separately exposed:
power-flow root convergence is not an OPF feasibility certificate.

The result retains native root diagnostics, generated powers, both branch terminal
powers, branch losses, shunt power, bus balance and total complex-power balance.
An unexecuted mode proposal is never labeled as the returned solved mode.

### Fixed-mode derivatives

Compile topology and choose modes outside JAX transformations. Then differentiate
`fixed_mode_power_flow(compiled, injections, modes=...)`. The native matrix-free
implicit-root tangent/adjoint path differentiates the root, not the Newton trace.
For a saturated mode the supplied reactive injection must include its bound.

```python
import jax
import jax.numpy as jnp

def voltage_imaginary(load):
    specified = compiled.specified_power.at[1].set(-load + 0j)
    return power.fixed_mode_power_flow(compiled, specified).voltage[1].imag

dv_imag_dp = jax.grad(voltage_imaginary)(jnp.asarray(0.5))
```

The analytic derivative here is −0.1. Derivatives require a successful, nonsingular
native root. No differentiability is claimed across active-set or topology changes;
a fixed-mode derivative at a switching boundary is only a one-mode derivative.

## DC flow and optimal power flow

`solve_dc_power_flow` uses a native equality-constrained `LinearProgram` for the
lossless unit-voltage approximation. A branch carries
Pfrom = (theta_from − theta_to − phase)/(x × tap), with Pto = −Pfrom.
Shunt conductance is evaluated at unit voltage. Resistance loss, reactive balance,
voltage magnitudes and charging are not modeled. The explicit approximation label
on every DC result prevents interpreting it as original AC feasibility. These are
the published [DC branch and phase-shift
equations](https://matpower.org/docs/ref/matpower7.1/lib/makeBdc.html).

`compile_dc_opf` returns a domain compilation containing a native `LinearProgram`
for linear costs or `QuadraticProgram` for convex quadratic costs. It enforces
every bus's balance, one reference angle per island, generator active-power limits
and signed branch-flow limits. `Generator.cost=(c2,c1,c0)` means
c2 × Ppu² + c1 × Ppu + c0; it is not implicitly an MW polynomial.
`solve_dc_opf` retains the native primal/dual result and independently recomputes
original DC balance, limits and objective.

`compile_ac_opf` builds a `StructuredNonlinearProgram` with exact native sparse
Jacobian and Hessian plans declared from network topology. Coordinates are
rectangular voltage plus generator P/Q. Reference angles, fixed generator values
and offline generator values are eliminated; reference voltage magnitudes remain
optimization variables. Constraints are original P/Q balance, voltage magnitude
bounds, generator P/Q bounds, and **both** terminal |S|² limits. Nothing is softened,
and no shedding/penalty dispatch is added. `solve_ac_opf` defaults to native
`PrimalDualInteriorPoint(mode="sparse-augmented")` and accepts an explicit native
structured method/termination. An optional `operating_point` provides the initial
coordinates without changing the constraints.

```python
congested = power.PowerNetwork(
    (power.Bus("r"), power.Bus("d")),
    (power.Branch("rd", "r", "d", 0, 0.1, rate=0.4),),
    (
        power.Generator("cheap", "r", p=0.3, p_min=0, p_max=2,
                        q_min=-1, q_max=1, cost=(0.1, 1, 0)),
        power.Generator("local", "d", p=0.7, p_min=0, p_max=2,
                        q_min=-1, q_max=1, cost=(0.1, 3, 0)),
    ),
    (power.Load("demand", "d", 1, 0.1),),
)
opf_study = power.PowerStudy((power.BusControl("r", "reference"), power.BusControl("d")))
dc = power.solve_dc_opf(congested, study=opf_study)
ac = power.solve_ac_opf(congested, study=opf_study)
print(dc.converged, dc.original_feasibility)
print(ac.converged, ac.original_feasibility, ac.native_result.optimization.status)
```

AC results recompute the original electrical equations, either-end MVA and
generator/voltage bounds. Success requires native optimization success **and**
finite original feasibility within the requested tolerance. AC OPF is nonconvex:
this is a local-candidate certificate, not a claim of a global optimum.

## Machine/network DAEs and events

`initialize_power_dynamics` and `initialize_smib` derive machine states and controller
references from a converged power-flow operating point and call native
`solver.initialize_dae`. The result distinguishes an equilibrium from mere DAE
consistency. Every online generator requires a machine or explicit membership in
`infinite_buses`; a power-flow reference label does not automatically become an
infinite source. Generator-free external-reference injection must also be explicitly
covered. Multiple machines may share a bus. Source-free dynamic islands fail.

The default `load_model="constant_power"` preserves the same PQ demand law as power
flow: native algebraic KCL contains conj(Sload/V), without an epsilon voltage floor
or impedance fallback. Loaded-bus voltage collapse can therefore cause genuine
nonfinite/root/consistency failure. The separately requested
`load_model="constant_impedance"` freezes conj(Sload)/|Voperating|² and changes the
physical load law deliberately. The selected fidelity is recorded in
`initialization.model.load_model` and retained by each dynamics result.

The closed machine family uses quasi-steady stator/network equations and no
magnetic saturation. In rotor coordinates, V exp(−i delta) = Vq − i Vd and generated
current I exp(−i delta) = Iq − i Id. The generated current is the negative of the
machine's inward terminal current. Speed is pu deviation from synchronous speed:

- delta_dot = 2 pi frequency × speed_deviation.
- 2 H × speed_deviation_dot = Pmechanical − Pelectrical − D × speed_deviation.
- `ClassicalMachine` has a fixed internal EMF behind R+i Xd_prime.
- `Order4Machine` adds Eq_prime and Ed_prime:
  - Td0_prime × Eq_prime_dot = Efield − Eq_prime − (Xd−Xd_prime) Id.
  - Tq0_prime × Ed_prime_dot = −Ed_prime + (Xq−Xq_prime) Iq.
- Pelectrical includes terminal active power and stator copper loss on machine base.
- `FixedExciter` and `FixedGovernor` hold operating-point field/mechanical values.
- `FirstOrderAVR` tracks clip(gain × (Vreference−|V|), lower, upper) with its declared
  time constant.
- `DroopGovernor` tracks clip(Pmechanical_reference−speed_deviation/droop, lower,
  upper) with its declared time constant.

Controller limits apply to the command; they are not undocumented hard state
projections. The fourth-order model is **not** GENROU and has no subtransient states.
The precise equations and state labels are defined in `_dynamics.py` through the
native `DifferentialAlgebraicSystem` and `DAEInitializationSpec.index_one()`.

```python
smib_network = power.PowerNetwork(
    (power.Bus("infinite"), power.Bus("machine")),
    (power.Branch("tie", "infinite", "machine", 0, 0.2),),
    (power.Generator("unit", "machine", p=0.6, q_min=-1, q_max=1),),
)
smib_study = power.PowerStudy((power.BusControl("infinite", "reference"),
                                power.BusControl("machine", "pv")))
smib_compiled = power.compile_network(smib_network, smib_study)
smib_pf = power.solve_power_flow(smib_compiled)
initialized = power.initialize_smib(
    smib_compiled, smib_pf,
    power.ClassicalMachine("unit", inertia=4, damping=1, xd_prime=0.3),
    infinite_bus="infinite",
)
requested = jnp.linspace(0, 0.2, 41)
trajectory = power.simulate_power_dynamics(
    initialized, requested,
    events=(power.PowerEvent(float(requested[10]), "fault", "machine", admittance=2-5j),
            power.PowerEvent(float(requested[20]), "clear", "machine")),
)
print(initialized.valid, trajectory.valid, trajectory.status)
```

Supported events are finite balanced bus-shunt `fault`, matching bus `clear`, and
branch breaker `trip`/`reclose`. A fault must have finite nonzero passive admittance;
a bolted fault is rejected, not approximated by an undocumented large number.
Simultaneous events execute in caller tuple order. Invalid target/state transitions
and source-free islands produce explicit event failure evidence.

Execution is segmented at exact event times and uses **native** `solve_dae`,
`DAEResetMap`, consistency candidates and scheduled hybrid event evidence. There is
no power-specific integrator. Differential states remain continuous; algebraic
voltages and rates are reconstructed consistently. Each changed-topology segment
starts with fresh native BDF history. The default native adaptive BDF policy lands
on every requested sample and event time, with no sample merging. Explicit caller
policies remain unchanged, including fixed-grid adjacent-step restrictions.
The default stage nonlinear target is the preceding representable value below
`min(residual_tolerance, constraint_tolerance * sqrt(N_algebraic / N_total))`.
This follows from `RMS_algebraic <= sqrt(N_total / N_algebraic) * RMS_all`:
convergence of the all-equation root must also satisfy the separately normalized
algebraic constraint certificate. Native physical acceptance tolerances are
unchanged; only the default inner root is solved to the required accuracy.
Events intended on sample nodes should take their times from those actual nodes,
as above: independently rounded decimal times can denote distinct nearby instants.
Extremely close distinct inputs remain distinct and can exceed native numerical
resolution, in which case failure remains explicit. Samples never interpolate
across a topology change. `PowerEventEvidence` exposes before/after states and residuals, differential
jumps, candidate consistency, scheduled evidence, application status and restart
order. A rejected candidate is never adopted; subsequent work is `not_run`.
The native scheduled tape covers differential coordinates only and does not claim
an algebraic projection Jacobian or event-time sensitivity.

## Nonexecuting import boundaries

Each parser returns `PowerCaseAdaptation(network, study, report, dynamics)`, using the
shared `interchange.AdapterReport`. A failed conversion raises `PowerImportError`
with its invalid `.report`. Numeric, row, token, total UTF-8 byte and XML-depth
bounds are controlled by `PowerParserLimits`. Parsers accept resident text, not
paths or executable objects. They do not invoke MATLAB/PSS/E, execute statements,
fetch resources or fabricate missing profiles.

### MATPOWER version 2

```python
case_text = """function mpc = analytic_two_bus
mpc.version = '2';
mpc.baseMVA = 50;
mpc.bus = [1 3 0 0 0 0 1 1 0 110 1 1.1 .9;
           2 1 10 0 0 0 1 1 0 110 1 1.1 .9;];
mpc.gen = [1 10 0 50 -50 1 50 1 50 0;];
mpc.branch = [1 2 0 .1 0 100 0 0 0 0 1 -360 360;];
mpc.gencost = [2 0 0 3 .01 2 3;];
end
"""
adapted = power.parse_matpower(case_text)
parsed_flow = power.solve_power_flow(adapted.network, study=adapted.study)
print(adapted.report.status, parsed_flow.status)
```

The grammar accepts only numeric version/baseMVA/bus/gen/branch/gencost assignments
inside a function-case wrapper. Total MW/Mvar quantities become pu; degrees become
radians; bus shunts and tap phase shifts preserve their signs. An MW quadratic
cost is rescaled to the canonical pu polynomial. Active capability curves, angle
limits, automatic controls, piecewise/reactive/startup/shutdown costs and isolated
buses are rejected. RATE_A is enforced; discarded B/C ratings and area/zone metadata
are explicitly reported. MATPOWER has no frequency, so the recorded import
assumption is 60 Hz. A validated adaptation is not necessarily lossless: inspect
its report. The format is documented by the [official caseformat
reference](https://matpower.org/docs/ref/matpower7.1/lib/caseformat.html).

Physical equipment is returned in `.network`, while parsed bus modes and operating
voltage metadata are returned in `.study`. Active controlled-generator VG overrides
the initial bus VM with explicit report evidence; conflicting same-bus active VG
setpoints fail instead of selecting one silently. The physical network can be
reused with a different explicit study without rewriting equipment.

### Paired RAW/DYR and CGMES

`parse_psse(raw_text, dyr_text)` admits complete **RAW revision 33** with
constant-power loads, fixed shunts, branches and fixed two-winding transformers
with CW=CZ=CM=1. Every online generator requires exactly one positive-inertia
`GENCLS` DYR record. RAW MBASE/ZR/ZX and DYR H/D become a real `ClassicalMachine`,
not a profile name. Missing models or unsupported active models fail. Remote
regulation, ZIP loads, switched shunts, DC/FACTS, three-winding and automatic tap
semantics fail. The RAW subset admits one active generator per bus rather than
discarding RMPCT sharing, and rejects active area-interchange/slack controls and
transformer magnetizing/correction/control fields. Inactive discarded models are
audited. The included
`tests/unit/applications/power/test_interchange.py` contains complete paired `RAW`
and `DYR` fixtures, including both winding ratios and an inactive unknown model.

`parse_cgmes(text_or_documents, base_mva=100, frequency=50)` is the explicit
**CGMES 2.4.15 / CIM16** balanced bus-branch subset. EQ-Core, TP and SSH profiles are
required; optional SV provides voltage kV/degrees. RDF links and exact namespace /
profile identifiers are checked locally. The subset includes lines, constant-power
consumers, synchronous generators and fixed linear shunts. CIM inward equipment
P/Q is negated for generators; line ohms/siemens are converted on the node kV base.
Transformers, switches, connectivity-node topology, unknown classes/properties and
controls other than local generator voltage regulation are rejected. No full-CGMES
conformance or topology processing is claimed. The same regression file contains a
complete real RDF/XML `CGMES` fixture with linked equipment, terminal and node
resources; it is not a fabricated profile-name success case.

```python
from pathlib import Path

# These are actual user-supplied files; the parsers themselves perform no I/O.
paired = power.parse_psse(Path("case.raw").read_text(), Path("case.dyr").read_text())
paired_compiled = power.compile_network(paired.network, paired.study)
paired_pf = power.solve_power_flow(paired_compiled)
paired_dynamics = power.initialize_power_dynamics(paired_compiled, paired_pf, paired.dynamics)

cgmes = power.parse_cgmes(tuple(Path(name).read_text() for name in ("EQ.xml", "TP.xml", "SSH.xml")))
```

Optional OpenDSS execution is a separate external-runtime boundary. Its real engine
status, units, signs, balanced qualification and artifacts must be inspected before
comparison. A successful external process is not proof of a matching balanced
network or native model fidelity, and no external engine is used as a hidden native
solver fallback.

## Reproducible evidence and benchmarks

The focused power regressions cover analytic two/three-bus signs and losses, bases,
island references, Q-limit transitions/reference failure, fixed-mode implicit
sensitivity, constrained native OPF, machine initialization, event state continuity,
failure evidence and real fail-closed parsers.

`python tools/power_benchmarks.py` uses the existing synchronized/repeated benchmark
helpers. It reports separate compilation/first/warm timings, dtype/backend, native
nonlinear work and residual evidence, followed by constrained native DC and AC OPF
original-feasibility evidence. Timing rows retain failure statuses; they do not turn
an unfinished solve into a benchmark success.
