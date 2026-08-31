# Field–circuit and electrothermal coupling

`FieldPortModel` retains typed field-solver wave ports and an optional causal descriptor realization. Frequency-only field responses remain frequency-only until an explicit causal macromodel is supplied. Basis identity, normalization, orientation, material identity, and reference plane remain part of the port contract.

`prepare_electrothermal_circuit` augments a prepared circuit DAE with one differential temperature state. The caller supplies an explicit `heat_power(time, circuit_state, temperature, args)` function. The monolithic residual contains the circuit residual and

```text
heat_capacity * temperature_rate
- heat_power
+ thermal_conductance * (temperature - ambient_temperature)
```

Temperature is passed into circuit element arguments, allowing temperature-dependent laws without a separate co-simulation API. Diagnostics report circuit residual, thermal residual, supplied heat, ambient loss, and finiteness.

No one-pass explicit field/circuit or circuit/thermal exchange is inferred stable. Partitioned waveform relaxation, spatial thermal PDE coupling, and time-domain Maxwell exchange require explicit interface iteration and energy-defect evidence.
