# Thermofluid processes

`phydrax.applications.thermofluids` provides typed thermofluid declarations that lower to the native acausal DAE substrate. It does not introduce another graph, structural analyzer, nonlinear solver, or DAE runtime.

`ThermofluidPortSpec` records physical port kind, flow direction, catalog identity, homogeneous thermodynamic model, and state-pair convention. `ThermofluidProcessPlan` validates each connection before producing `AcausalDAESource` and a separate `process_model_id` that includes physical component parameters.

Initial material links are fixed-direction and pairwise. A material connection requires one inlet and one outlet with identical catalog, thermodynamic model, and state-pair identity. Multiway material junctions require explicit mixer/splitter components; generic potential/flow connection semantics do not define advected enthalpy or composition mixing.

`fixed_material_boundary_component` and `isenthalpic_valve_component` provide a minimal single-stream network. Heat, scalar-power, and shaft ports are distinct kinds and cannot connect to material ports.

## Compressor maps

`CompressorMapPlan` stores corrected-speed and operating-line axes plus corrected-flow, pressure-ratio, and isentropic-efficiency tables. It uses native rectilinear interpolation and returns explicit support. Off-map points are unsupported; constant-boundary interpolation is never treated as physical extrapolation.

`CompressorPlan.design` creates an immutable calibration artifact tied to its exact map. Compressor evaluation solves the isentropic and actual outlet states through the selected homogeneous thermodynamic model, conserves mass flow, and reports shaft power. Total gas stations remain distinct from generic process ports.
