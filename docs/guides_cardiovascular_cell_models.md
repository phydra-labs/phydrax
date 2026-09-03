# Cardiovascular ventricular cell models

PhydraX provides two model-specific ventricular reaction systems:

- `TenTusscherPanfilov2006Model`: the 19-state human ventricular TP06 system,
  including sodium, calcium, transient-outward, delayed-rectifier,
  inward-rectifier, exchanger, pump, plateau, and background currents;
- `ORdVentricularModel`: the 41-state O'Hara--Rudy dynamic system, including
  cleft/cytosol ion pools, network/junctional SR pools, CaMK-dependent current
  fractions, release-state dynamics, and a decomposed electrogenic Na/K pump.

These are independent mathematical implementations. They do not load or
translate external simulator code or model files.

## Units and signs

The cardiovascular kernel uses mm, ms, mg, mV, kPa, and mm³. Cell-model
concentrations are mM. Membrane currents are physical surface densities in
µA/mm² and use an **outward-positive** convention. Therefore an outward ionic
current lowers voltage:

```text
dV/dt = -(I_ion + I_outward_applied) / C_m
```

`CardiacMembraneScaling` connects the cellular and tissue descriptions. With
surface-to-volume ratio `chi` in mm⁻¹ and membrane capacitance `C_m` in
µF/mm²,

```text
I_volume = chi I_surface
D = conductivity / (chi C_m)
dV/dt_conduction = div(conductivity grad(V)) / (chi C_m)
```

The default `chi = 140 mm⁻¹` and `C_m = 0.01 µF/mm²` correspond to
1400 cm⁻¹ and 1 µF/cm². The SI conversion factors are explicit: mS/mm and S/m,
µA/mm² and A/m², and mV/ms and V/s have equal numeric values; µA/mm³ to A/m³
has factor 1000.

Tissue stimuli accepted by `PreparedReaction.rates` are instead
**inward-positive** µA/mm³, which is the common pacing convention.

## Typed fidelity and phenotype routes

Select a transmural phenotype with `VentricularCellPhenotype`, not a runtime
string:

```python
import jax.numpy as jnp

from phydrax.applications.cardiovascular import electrophysiology as ep

model = ep.TenTusscherPanfilov2006Model(
    phenotype=ep.VentricularCellPhenotype.EPICARDIAL,
)
state = model.initialize((256,), dtype=jnp.float64)
evaluation = model.evaluate(state)
```

Endocardial, midmyocardial, and epicardial instances have different stable
`model_id` values. A string such as `"epicardial"` is rejected rather than
silently interpreted as a fidelity route.

## Model-specific named state

Each class owns an immutable `CardiacReactionStateLayout`. The final state axis
is the model's named channel axis, while every leading axis is an independent
cell. TP06 has 19 channels and ORd has 41; neither is padded to a universal
state size.

```python
layout = model.state_layout
print(layout.state_names)
print(layout.gate_names)
print(layout.concentration_names)

calcium_i = layout.channel(state, "calcium_i_mM")
named_channels = layout.unpack(state)       # host-inspectable SoA views
round_trip = layout.pack(named_channels)
```

The companion `CardiacReactionParameterLayout` pins parameter order and units.
`model.default_parameters` is the matching immutable numeric vector. A
replacement parameter array must have the exact declared final-axis size;
leading axes may batch parameter sets with cell axes.

## Pure evaluation and exact gates

`evaluate` is a pure, vectorized operation. It returns
`CardiacReactionEvaluation` with:

- `state_rate`, in each channel's unit per ms;
- `gate_steady_state` and `gate_time_constant_ms`;
- `current_density_uA_per_mm2` and its fixed `current_names` decomposition;
- `total_outward_current_uA_per_mm2`;
- `calcium_cytosol_mM`, `calcium_cytosol_rate_mM_per_ms`,
  `calcium_sr_flux_mM_per_ms`, and
  `calcium_membrane_current_uA_per_mm2`;
- `charge_balance_residual_uA_per_mm2`, `valid`, and the static `model_id`.

```python
rates = model.rates(state)
gated = model.exact_gate_update(state, dt_ms=jnp.asarray(0.02))
ina = evaluation.current("I_Na")
ica = evaluation.current("I_CaL")
```

`exact_gate_update` applies the analytic Rush--Larsen update
`x_inf + (x - x_inf) exp(-dt/tau)` to every declared first-order gate. It
preserves voltage and concentration channels. A tissue or cell integrator must
advance those non-gate channels using its declared ODE scheme; the gate helper
does not hide an Euler concentration update.

The calcium fields are live model outputs suitable for active-tension coupling.
They are not reconstructed from voltage or replaced by a prescribed transient.

## Fixed homogeneous tissue blocks

Plan and prepare a fixed-size block before tissue stepping:

```python
plan = ep.plan_reaction(model, node_count=256, dtype=jnp.float64)
reaction = ep.prepare_reaction(plan)
voltage_mV, local_state = reaction.initialize()

dv_dt, dlocal_dt = reaction.rates(
    voltage_mV,
    local_state,
    stimulus_uA_per_mm3=jnp.zeros((256,)),
)
outward_ionic_uA_per_mm3 = reaction.currents(voltage_mV, local_state)
```

The prepared object pins `plan_id`, `model_id`, node count, dtype, state count,
and gate count. Its split `local_state` contains all non-voltage channels,
including concentrations; `true_gate_count` records how many channels receive
an exact first-order update. Regional tissue assignment should route separate
homogeneous worksets rather than pad different cell families into one array.

## Domain handling and evidence

Positive ion concentrations, finite parameters and states, normalized bounded
gates, and a physiological voltage domain are admission requirements.
`validate_state` is the host-side boundary check and raises on invalid input.
Inside compiled evaluation, invalid cells fail closed: `valid` is false and
rates/currents are NaN instead of a plausible clipped trajectory.

The GHK current expressions use analytic series at their removable singular
voltages (15 mV for the TP06 shifted L-type expression and 0 mV for ORd).
Values and derivatives are finite at those exact voltages; no epsilon voltage
shift changes the equations.

Charge evidence includes the applied outward surface current when present:

```text
C_m dV/dt + I_ion + I_outward_applied = charge_balance_residual
```

A valid direct reaction evaluation closes this residual to floating-point
roundoff. Concentration and SR-flux fields provide separate calcium exchange
evidence rather than overloading the membrane charge residual.

## Inspectable pinned reaction algebra

`PinnedReactionIR` is a host-only, closed algebraic IR for declared expressions.
Inputs are positional slots, operations are enums, and outputs have fixed names.
`compile_reaction_ir` lowers the tree to immutable register instructions.
Neither construction nor execution uses `eval`, runtime source strings,
plugins, or user callables.

```python
from phydrax.applications.cardiovascular.electrophysiology import (
    PinnedReactionIR,
    ReactionBinaryOperator,
    ReactionIRBinary,
    ReactionIRInput,
    ReactionIROutput,
    compile_reaction_ir,
)

voltage = ReactionIRInput(0)
reversal = ReactionIRInput(1)
ir = PinnedReactionIR(
    "driving-force-v1",
    ("voltage_mV", "reversal_mV"),
    (
        ReactionIROutput(
            "driving_force_mV",
            ReactionIRBinary(
                ReactionBinaryOperator.SUBTRACT,
                voltage,
                reversal,
            ),
        ),
    ),
)
compiled = compile_reaction_ir(ir)
(driving_force,) = compiled((jnp.asarray(-80.0), jnp.asarray(-90.0)))
print(compiled.program_id, compiled.inspect())
```

`interpret_reaction_ir` supplies a separate direct-tree execution route for
qualification against the lowered register program. The ventricular models
also expose a pinned ohmic-current IR through `model.reaction_ir` for inspection
and N-version checks.
