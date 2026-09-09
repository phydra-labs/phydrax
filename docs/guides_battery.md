# Thermal equivalent-circuit batteries

`phydrax.applications.battery` exposes the lumped thermal equivalent-circuit
model (ECM) driven by prescribed terminal current and rest steps. API
availability is not release authorization: production execution requires
current authenticated evidence for the exact distribution. Historical evidence
from another source build does not authorize this build. The ECM equation
support tuple is `battery.THERMAL_ECM_SUPPORT`:

| coordinate | supported value |
| --- | --- |
| capability | `battery.simulation` |
| `model_id` | `battery:ecm:thermal-prescribed-current` |
| `equation_form` | `ode` |
| `control` | `prescribed-current` |
| `circuit_connected` | `false` |

The support boundary is conjunctive. Changing any coordinate describes a
different, unqualified route.

## Passive terminal convention and SI units

`battery.PASSIVE_TERMINAL_CONVENTION` is used by every public protocol and
result:

- Terminal voltage is `V = V_positive - V_negative` in volts (V).
- Terminal current `I` is positive when it enters the positive terminal, in
  amperes (A).
- Absorbed terminal power is `P_abs = V I` in watts (W). Consequently,
  discharge has `I < 0`; positive power delivered by the cell is `-V I`.
- Stored charge `q` is in coulombs (C) and obeys `dq/dt = I`. Positive current
  increases stored charge.
- Temperature is absolute temperature `T` in kelvin (K), resistance is in ohms
  (Ω), capacitance is in farads (F), heat capacity is in joules per kelvin
  (J/K), and thermal conductance is in watts per kelvin (W/K).

Do not substitute the common battery convention “positive current means
discharge.” Negate such input data before constructing `BatteryProtocolValues`.

## Exact equations

For fixed positive series resistance `R_0`, fixed positive branch resistances
`R_j`, fixed positive branch capacitances `C_j`, fixed positive reference
capacity `Q_ref`, and fixed positive lumped heat capacity `C_th`, the state is
stored charge `q`, RC polarization voltage `v_j` for every branch, and cell
temperature `T`. State of charge is

```text
z = q / Q_ref
```

The bounded reference open-circuit-voltage and entropic laws are
`U_ref(z)` and `dU/dT(z)`. They must have identical finite SOC support. The
model evaluates

```text
U(z, T) = U_ref(z) + (T - T_ref) dU/dT(z)
V        = U(z, T) + I R_0 + sum_j v_j

dq/dt   = I
dv_j/dt = -v_j / (R_j C_j) + I / C_j

Q_irr  = I² R_0 + sum_j v_j² / R_j
Q_rev  = I T dU/dT(z)
Q_cool = G (T - T_amb)
C_th dT/dt = Q_irr + Q_rev - Q_cool
```

Here `G > 0` is the outward thermal conductance and `T_amb > 0` is the ambient
temperature. `ThermalEquivalentCircuitParameters` requires every resistance,
capacitance, capacity, heat capacity, conductance, and absolute temperature to
be positive. The released property choices are
`ConstantPropertyLaw` and `TabulatedPropertyLaw`; both return an explicit
in-support flag and never extrapolate beyond their declared SOC interval.
The model has ideal coulombic efficiency, fixed capacity, no hysteresis, and no
fade state.

## Protocols, controls, and stopping

A `BatteryProtocolPlan` is an immutable ordered tuple of:

- `CurrentStepPlan(duration_s)`, whose one dynamic amplitude is supplied in
  amperes through `BatteryProtocolValues`; or
- `RestStepPlan(duration_s)`, which supplies exactly zero current and consumes
  no amplitude.

Durations are finite and positive. The protocol owns an explicit left- or
right-sided observation convention at transition nodes. Execution separately
uses the forward right limit, and preparation forces every current/rest
transition into the native integration topology. There is no hidden ramp,
feedback law, voltage control, power control, or current interpolation.

`VoltageStopGuard`, `CurrentStopGuard`, and `TemperatureStopGuard` provide
optional lower (`"below"`) or upper (`"above"`) terminal conditions. Dynamic
thresholds appear in guard order in `BatteryProtocolValues.stop_thresholds`.
A guard stops the native solve; it does not clamp the state or replace failed
samples.

The thermal ECM exposes these selectable observables with SI units:

| name | unit |
| --- | --- |
| `voltage_v` | V |
| `current_a` | A |
| `temperature_k` | K |
| `state_of_charge` | 1 |
| `charge_c` | C |
| `polarization_voltage_sum_v` | V |
| `terminal_power_w` | W |
| `irreversible_heat_w` | W |
| `reversible_heat_w` | W |
| `cooling_power_w` | W |
| `net_heat_w` | W |

`BatteryOutputPlan` selects an ordered, duplicate-free subset. The result keeps
that order and its units.

## Status, termination, and ledgers

`BatteryExperimentResult.application_status` is a scalar
`BatteryRunStatus` code:

| status | meaning |
| --- | --- |
| `SUCCESS` | native solve, model domain, selected outputs, and model ledger are valid |
| `DOMAIN_ERROR` | SOC, temperature, or a bounded property evaluation left its domain |
| `NATIVE_SOLVE_FAILED` | the native ODE or DAE solve did not report success |
| `NONFINITE_OUTPUT` | a selected observation was not finite |
| `CAPACITY_EXCEEDED` | reserved fail-closed capacity disposition |
| `MODEL_LEDGER_FAILED` | native execution completed but the model ledger failed |

Require `result.successful` and `result.ledger.successful`, and consume only
samples marked valid. A successful terminal event may leave an invalid saved
suffix; require every sample valid only when the complete requested horizon is
needed. Array presence alone is not success. `result.termination.reason` is
`"completed"` for normal completion. A guard records the winning reason, time,
and protocol step. Initial equality or an already-violated stop condition stops
immediately; its `derivative_valid` flag is false rather than claiming a
crossing-time derivative.

`ThermalEquivalentCircuitLedger` records integrated terminal charge, charge
defect, RC-energy change, resistive dissipation, thermal-energy change,
irreversible and reversible heat, ambient heat loss, and thermal defect. Its
`successful` flag establishes only finite, ordered, internally computable
ledger evidence. Applications must impose tolerances appropriate to their own
numerical plan rather than interpreting the flag as a physical-accuracy or
safety certificate.

## Release evidence and requalification

`battery.THERMAL_ECM_CANDIDATE` is the immutable declaration used by the
qualification campaign. It is deliberately unreleased and contains no evidence
IDs or timestamps. Production release must be reconstructed from complete,
externally retained evidence, not opaque evidence IDs.

All timestamps stored in these records and their campaign specifications are
UTC Unix-epoch nanoseconds. Monotonic-clock values are used only for
non-persisted in-process ordering checks and cannot appear in an evidence
bundle.

Production admission requires the complete typed criterion/start/observation/
evidence chain, model-specific validity envelope, reference and rights records,
resource criteria, independent clean-execution comparison, and complete passing
coverage. The release builder constructs the gate internally and limits its
expiry to the earliest cited expiry. Caller-supplied gate names or content hashes
alone are not proof.

Criterion approval, execution, scientific review, entry decisions, release, and
channel promotion have distinct asymmetric signing roles. Deployment trust
contains public verification keys with activation, expiry, and revocation
policy. A signed release index is necessary but not sufficient: retained typed
proofs, current dependency profiles, and exact distribution/support/envelope
bindings must also validate.

Preparation with an explicit candidate profile is development execution.
Preparation with a released profile requires authenticated
`BatteryExecutionAdmission` and the exact `distribution_id`. Results distinguish
numerical and predictive admission IDs. Numerical admission establishes the
declared equations and numerical envelope; predicting a particular cell needs
its separately authenticated parameter/data/uncertainty profile.

Historical JSON files under `benchmarks/battery/evidence` document their original
execution only. Do not load their issue time as the current validation time or
reuse them after changing source, a harness, or a distribution.

The scientific evidence declares requalification triggers. A backend, source
build, device, discretization, environment, parameter set, precision, support,
or topology change requires replacement evidence and a new externally reviewed
release gate. Expiration also requires new current evidence; changing a JSON ID
or copying an old gate is not requalification.

## Simulation and generic optimization

The repository example is deterministic and network-free:

```text
python examples/battery_simulation_and_optimization.py --development
```

The explicit flag selects the unreleased candidate and prints that disposition.
The example executes a current/rest experiment and checks status, validity,
conservation, and domain margins. It composes the prepared experiment with
`phydrax.optim.MinimizationProblem` and a bounded generic optimizer to choose one
constant current amplitude for a target final charge. It re-executes the accepted
amplitude and checks status, bounds, target, optimality, and physical-domain
margins. This local re-execution is not authenticated independent release replay.
The optimization demonstration is development-only. A numerical-model token
does not authorize a transformed optimization workflow; production numerical
execution is illustrated by the host-dispatched circuit ECM and SPMe examples.
This is ordinary user composition with the generic optimization API, not a
battery-specific protocol-design or current-control capability.

## Nonclaims

This surface is a cell-level lumped model for the exact equations and support
tuple above. It does not provide circuit or multi-cell assembly, detailed
porous-electrode or full-order electrochemistry, parameter estimation,
calibration, experimental design, degradation or side-reaction prediction,
state estimation, or automatic control. It makes no pack, safety, abuse,
thermal-runaway, lifetime, fast-charge, warranty, certification, regulatory, or
commercial-readiness claim. A successful run or released scientific profile
must not be interpreted as evidence for any of those uses.
