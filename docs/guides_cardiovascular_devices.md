# Cardiovascular vessels, devices, and oxygen transport

PhydraX models cardiovascular device hydraulics in the kernel unit system: mm,
ms, mg, mV, kPa, and mm³. Flow is positive from each component's `inlet` to
its `outlet`. Device maps and physiological constitutive laws have declared,
closed validity intervals. Evaluation outside one of those intervals returns
unsuccessful evidence and does not extrapolate or divide by an artificial
epsilon.

## Conservative 1D vessels

`SquareRootTubeLaw` is the named `square-root-elastic` law

```text
p(A) = p₀ + β (sqrt(A/A₀) - 1).
```

Its plan records the reference area and pressure, stiffness, and minimum and
maximum admissible area ratios. For blood density `ρ`, the local wave speed and
pressure/flow characteristic impedance are

```text
c² = A (dp/dA) / ρ,       Z = ρ c / A.
```

`Vascular1DPlan.prepare` fixes the grid and the tube law before integration.
The runtime advances conservative area/flow cell averages,

```text
∂A/∂t + ∂Q/∂x = 0,
∂Q/∂t + ∂[Q²/A + Ψ(A)]/∂x = -8πμQ/(ρA),
```

where `dΨ/dA = A(dp/dA)/ρ`. A local Lax–Friedrichs interface flux preserves
the finite-volume balance. `step_vascular_1d` returns both the candidate and
accepted state, boundary fluxes, mass and momentum balance residuals, CFL
number, and status. An invalid area, boundary, non-finite value, or excessive
CFL number leaves the accepted state unchanged.

```python
import jax.numpy as jnp
from phydrax.applications import cardiovascular as cv

law = cv.SquareRootTubeLaw(
    100.0,
    10.0,
    reference_pressure_kPa=8.0,
    minimum_area_ratio=0.5,
    maximum_area_ratio=2.0,
)
runtime = cv.Vascular1DPlan("aorta", 128, 256.0, 0.02).prepare(law)
state = cv.initialize_vascular_state(
    runtime,
    jnp.full((128,), 100.0),
    jnp.zeros((128,)),
)
left = cv.VascularBoundaryState(state.area_mm2[0], state.flow_mm3_per_ms[0])
right = cv.VascularBoundaryState(state.area_mm2[-1], state.flow_mm3_per_ms[-1])
result = cv.step_vascular_1d(runtime, state, left, right)
assert result.evidence.successful
```

`CharacteristicTerminal` exposes the exact linear reflection coefficient
`(Zload - Zvessel)/(Zload + Zvessel)`. A matched load therefore has zero
reflection. `VascularJunctionPlan` fixes branch order; its solve enforces one
common junction pressure and exact signed flow conservation. `Vascular0DPort`
uses stable IDs such as `aorta.inlet`, and `couple_0d_pressure_port` turns a 0D
pressure into a characteristic-compatible ghost state. Cannulae, tubing,
hydraulic oxygenators, and fixed-speed pumps also lower to the canonical
`PressureFlowComponent` DAE ports instead of defining another network system.

## Pump maps and hydraulic components

`PumpHeadFlowMap` stores a rectangular speed × flow head table. Flow and speed
axes must be strictly increasing, head must be nonnegative and nonincreasing
with flow, and head must be nondecreasing with speed. In-domain evaluation is
bilinear. Out-of-domain flow or speed sets the appropriate `PumpMapStatus` bit,
returns `successful=False`, and returns no extrapolated head.

```python
pump = cv.PumpHeadFlowMap(
    "ecmo-pump",
    [0.0, 2.0, 4.0],                 # mm³/ms
    [2_000.0, 4_000.0],              # rpm
    [[6.0, 4.0, 2.0], [12.0, 8.0, 4.0]],  # kPa
)
point = cv.evaluate_pump_map(pump, 2.0, 3_000.0)
assert point.successful
```

`Cannula` and `TubingSegment` compute their linear resistance from length,
inner diameter, and dynamic viscosity. Their optional directional quadratic
loss represents entrances, bends, and connectors. `HydraulicOxygenator`
contains pressure-loss data only; its immutable `supports_gas_exchange` field
is always false.

`ECMOCircuitPlan` combines one map, drainage and return cannulae, zero or more
tubing segments, and one hydraulic oxygenator. `solve_ecmo_hydraulics` finds an
operating point only when pump head and the complete passive pressure load
bracket a root inside the map. Its evidence resolves the pressure balance by
component. A map refusal or unbracketed operating point cannot be reported as a
successful flow.

## Causal controllers and replay

`PacemakerControllerPlan` is an inhibited, lower-rate pacemaker with explicit
upper-rate, refractory, pulse-width, and amplitude limits. Its fixed state
contains only the last committed sample time, last accepted activation,
pulse end, paced count, and sample index. Non-monotone samples are rejected and
do not advance the time cursor. `replay_pacemaker_controller` calls the same
production transition for every recorded sample and returns fixed-shape event
and pulse arrays.

`PumpControllerPlan` provides a fixed-period PI flow controller. Integral,
speed, and slew bounds are part of the plan identity. The transition consumes
only the current measured flow and setpoint; a future record cannot affect an
earlier command. A timestamp that is noncausal or does not match the declared
sample period is rejected without committing the candidate. The event mask
records every active saturation and slew limit. Use
`replay_pump_controller` to deterministically reproduce a recorded command
sequence.

## Oxygen content and ECMO gas exchange

`BloodOxygenModel` represents both dissolved and hemoglobin-bound oxygen:

```text
Ctotal = α pO₂ + Hb · cbind · S,
S = pO₂ⁿ / (P50ⁿ + pO₂ⁿ).
```

`evaluate_oxygen_content` reports both components. Hill saturation has an
analytic inverse, while total content uses fixed-iteration monotone bisection
on the model's declared partial-pressure interval. Values outside that interval
are refused rather than clipped.

`mix_oxygen_content` weights content by nonnegative incoming volume flow and
reports the incoming/outgoing oxygen-flux residual. `OxygenTransportPlan` fixes
cell volumes and directed topology. Its upwind step transfers oxygen inventory
between cells exactly, reports the boundary inventory change and conservation
residual, and refuses a step when outgoing volume exceeds a cell's fixed
volume.

Gas exchange is a separate capability. Supplying only a
`HydraulicOxygenator` to `ECMOCircuitPlan` preserves inlet oxygen content and
sets both `gas_exchange_enabled` and `gas_exchange_performed` false. To enable
exchange, supply a `MembraneOxygenatorModel` with a `BloodOxygenModel`, gas
partial pressure, transfer capacity, and a positive flow validity interval.

```python
blood = cv.BloodOxygenModel(14.0)
exchange = cv.MembraneOxygenatorModel(
    blood,
    20.0,
    1.0,
    minimum_flow_mm3_per_ms=0.1,
    maximum_flow_mm3_per_ms=4.0,
)
plan = cv.ECMOCircuitPlan(
    pump,
    drainage,
    return_cannula,
    (tubing,),
    hydraulic_oxygenator,
    oxygen_model=exchange,
)
result = cv.run_ecmo_circuit(plan, 4_000.0, 1.0, 3.0, 12.0)
assert result.gas_exchange_enabled
assert result.gas_exchange_performed == result.successful
```

A hydraulic solve alone never claims oxygenation. Gas exchange is reported only
when the explicit membrane/content route succeeds at the solved positive flow.
