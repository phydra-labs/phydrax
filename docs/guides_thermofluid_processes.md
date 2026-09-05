# Thermofluid processes

`phydrax.applications.thermofluids` provides typed thermofluid declarations that lower to the native acausal DAE substrate. It does not introduce another graph, structural analyzer, nonlinear solver, or DAE runtime.

`ThermofluidPortSpec` records physical port kind, flow direction, catalog identity, homogeneous thermodynamic model, and state-pair convention. `ThermofluidProcessPlan` validates each connection before producing `AcausalDAESource` and a separate `process_model_id` that includes static component parameters and semantic device-law identities. Differentiable device coefficients remain numerical PyTree leaves, not host-fingerprinted constants.

Material links are fixed-direction and pairwise. A material connection requires one inlet and one outlet with identical catalog, thermodynamic model, and state-pair identity. A port may participate in at most one pairwise connection. Multiway material junctions require an explicit control volume: generic potential equality cannot define advected enthalpy or composition mixing.

## Signs and well-posed boundaries

Connection orientation is a physical port contract, not a universal electrical/phasor convention:

- Heat ports expose absolute temperature in kelvin and flow in watts. `HeatFlowOrientation.INTO_COMPONENT` makes positive flow enter its component; `OUT_OF_COMPONENT` makes positive flow leave. Connected temperatures agree and the two **inward** heat flows sum to zero.
- Material `mass_flow_orientation` is the sign mapping the native mass flow to flow entering its component. Existing boundary values remain source-outlet positive and sink-inlet negative (`mass_flow_orientation=-1`). A valve, mixer, or fluid exchanger retains inlet-positive/outlet-negative values (`mass_flow_orientation=+1`). Explicit orientation makes two successive valves conserve mass without flipping their downstream stream direction. This also preserves the original source/valve/sink values; a global `(1, -1)` on every link would not compose.
- Scalar-power and shaft ports remain separate kinds; no electrical phasor convention is imposed on them.

`material_boundary_component` prescribes only supplied fields. A flow/enthalpy source normally supplies `mass_flow` and `specific_enthalpy` while leaving pressure free. A pressure sink supplies `pressure` while leaving enthalpy and mass flow free. With `isenthalpic_valve_component`, the sink pressure and pressure ratio then determine the inlet pressure. Fully fixing both boundaries overdetermines this network. `fixed_material_boundary_component` remains available when all three values really are independently prescribed; it is not a well-posed source-and-sink pair for a valve.

`HeatPortBridge(orientation, temperature_offset=...)` converts an external temperature to Kelvin and external heat flow to inward watts. For Celsius use `temperature_offset=273.15`. This is an explicit boundary conversion, **not** an offset inside DAE potential equality. A building using heat-positive-into-zone consumes the bridge's `heat_into_component` value, not a raw outward heat-port variable.

## Solving heat exchange between two finite bodies

`thermal_capacitance_component` implements `C dT/dt = sum(Q_into)` with heat capacity in J/K. It takes differential initial temperatures, not prescribed-temperature equations. `thermal_conductor_component` implements `Q_left,in = G (T_left - T_right)` and `Q_left,in + Q_right,in = 0`, with conductance in W/K. The body accepts multiple independently connected ports through `port_count`; one port is named `heat`, or multiple ports are `heat_0`, `heat_1`, etc.

The following uses the existing structural compiler and DAE runtime, with no application-specific solver:

```python
import jax.numpy as jnp
import phydrax as phx

tf = phx.applications.thermofluids
hot = tf.thermal_capacitance_component("hot", heat_capacity=2.0)
cold = tf.thermal_capacitance_component("cold", heat_capacity=1.0)
link = tf.thermal_conductor_component("link", conductance=1.0)
process = tf.ThermofluidProcessPlan(
    (hot, cold, link),
    (
        tf.ThermofluidConnection("hot", "heat", "link", "left"),
        tf.ThermofluidConnection("link", "right", "cold", "heat"),
    ),
)
compiled = phx.dynamics.compile_acausal_dae(
    process.source, phx.dynamics.DAEStructuralPolicy(1, 0, tearing="none")
)
# Native compilation sorts variable names; pack these scalar unknowns by name.
initial_values = {"hot.temperature": 360.0, "cold.temperature": 280.0}
initial_state = jnp.asarray([
    initial_values.get(name, 0.0) for name in compiled.analysis.variable_names
])
# Fix body temperatures and algebraic rates, solving all algebraic values and
# the two initial temperature derivatives consistently.
problem = phx.solver.DifferentialAlgebraicProblem(
    compiled.system,
    initial_state,
    initialization=phx.solver.DAEInitializationSpec.from_masks(
        compiled.fixed_state_mask, compiled.fixed_rate_mask
    ),
    problem_id=process.process_model_id,
)
grid = phx.dynamics.TimeGrid(jnp.linspace(0.0, 0.5, 6), time_id="two-body")
solution = phx.solver.solve_dae(
    problem, grid,
    policy=phx.solver.DAESolvePolicy(method=phx.solver.BDFMethod(1)),
)
assert bool(solution.successful)
jet = compiled.reconstruction(solution.states[-1], solution.state_rates[-1])
energy = 2.0 * jet.value("hot.temperature") + jet.value("cold.temperature")
print(energy, jet.value("hot.temperature"), jet.value("cold.temperature"))
```

The energy relative to a common reference remains 1000 J. The hot body cools and the cold body warms. For this BDF1 grid, the temperature difference at step `n` is `80 / 1.15**n` K. Changing a port to outward-positive changes its reported flow sign, not this solution or its energy balance. Use `temperature_boundary_component(temperature=...)` for an infinite reservoir instead of a finite body; it fixes only temperature and solves the required heat exchange.

## Material mixing and fluid heat exchange

`material_mixer_component(inlet_count=..., species_count=..., catalog_id=..., thermodynamics_id=...)` is a steady adiabatic, zero-volume control volume with ports `inlet_0`, ... and `outlet`. It imposes:

- equal port pressures;
- total mass balance `sum(m_port) = 0`;
- advective enthalpy balance `sum(m_port * h_port) = 0`;
- one species mass balance `sum(m_port * Y_port,s) = 0` per species.

It never equates the different inlet enthalpies or mass fractions. Pairwise connection potentials propagate an individual stream's pressure, enthalpy, and composition only across that link. Source compositions must be normalized mass fractions in the shared catalog's order, with a common enthalpy reference. Outlet normalization follows from species and total mass balances; imposing another normalization equation would be redundant.

For example, source flows 2 and 1 kg/s with enthalpies 10 and 40 J/kg and mass fractions `(0.2, 0.8)` and `(0.8, 0.2)` yield outlet flow -3 kg/s, enthalpy 20 J/kg, and fractions `(0.4, 0.6)`. Only source enthalpy/flow/composition and sink pressure are prescribed. Positive total inflow is required for a unique mixture. Zero-flow mixing is singular, and reverse flow is outside this fixed-direction model; neither is repaired by clipping or a fallback outlet state.

Initialize with admissible approximate states, not all-ones vectors: use positive nominal pressures and temperatures within the provider's domain, direction-consistent flow signs, and normalized initial mass fractions. These are guesses, not extra prescribed equations or solved outlet values. Pack scalar guesses using the compiled variable-name order, as in the thermal example.

Small material networks combine raw pressure and enthalpy magnitudes around `1e5` with order-one mass/species equations. When requiring tight absolute conservation, select an explicit native initialization accuracy policy rather than weakening the physical residual checks. For the small examples, use `NewtonKrylov` with `LinearSolvePolicy(GMRES(restart=n), tolerance=TolerancePolicy(relative=1e-12, absolute=1e-13, max_steps=4*n))` and `NewtonForcingPolicy("constant")`, where `n` is the compiled scalar state dimension. Pass this as `DAESolvePolicy(initialization_method=...)` to `initialize_dae`. The full Krylov basis is appropriate for these small systems, not a default recommendation for large networks. Numerical convergence and the original unscaled mass/enthalpy/species residual audit remain separate acceptance checks.

`homogeneous_fluid_heat_exchanger_component(thermodynamics=..., mole_fraction=..., conductance=...)` adds a steady isobaric, perfectly mixed fluid control volume with `inlet`, `outlet`, and `heat` ports. Its thermal conductance relates the external heat-port temperature to the uniform outlet temperature. The native `HomogeneousHelmholtzPlan` simultaneously closes outlet pressure and mass-specific enthalpy from temperature, molar density, and fixed mole composition. Its energy equation is `m_in*h_in + m_out*h_out + Q_into = 0`. No constant-heat-capacity substitute, separate thermodynamic inversion solver, phase-transition assumption, or species-mixing shortcut is introduced.

After solving, evaluate the same provider at the reconstructed `temperature` and `molar_density`, with the declared `mole_fraction`, and require `state.evidence.successful`. Numerical DAE convergence certifies the residual equations, not thermodynamic support. The component is a single-phase fixed-composition stream; use an explicit mixer for changing composition.

## Heat pumps and resistance heating

The native array interface `HeatConversionLaw.evaluate(electrical_power, source_temperature, supply_temperature)` is shared directly with building simulation or planning. It returns `HeatConversionEvaluation` containing `electrical_power`, `delivered_heat`, `environment_heat`, and `successful`. It does not own another compiler or optimization framework.

- `ConstantCOPHeatPumpLaw(cop)` delivers `cop * electrical_power`; it explicitly extracts `(cop - 1) * electrical_power` from the environment.
- `ResistiveHeatingLaw(efficiency)` delivers `efficiency * electrical_power`; its negative `environment_heat` explicitly represents heat rejected as loss.

Both obey `delivered_heat = electrical_power + environment_heat`. Temperatures are absolute Kelvin and electrical input must be finite and nonnegative. For a positive temperature lift the heat-pump law also rejects a COP above the Carnot bound; it does not clip to a more favorable operating point. These are heating laws, not reversible refrigeration or arbitrary performance-map models.

COP and resistance efficiency are scalar JAX array leaves: the same laws work in `jax.jit`, calibration gradients, and Equinox model transforms. Constitutive identities describe the law, not a frozen numerical calibration. `successful` rechecks coefficient bounds during evaluation, including values changed by a model transform. Conversion DAE residuals retain the law and electrical input as native module leaves, rather than capturing them in opaque Python closures.

`heat_conversion_component(name, law=..., electrical_power=...)` lowers a law to an acausal component with `supply` and `environment` heat ports. Both raw flows are positive inward: the supply flow is negative when heat is delivered, while the environment flow is positive for heat extraction. Connect **both** terminals, e.g. a building body at the supply and a temperature reservoir at the environment. With 100 W electrical input and COP 3, the load receives 300 W and the environment loses 200 W. With resistance efficiency 0.8, the load receives 80 W and the environment receives 20 W.

The law's `successful` flag must be checked at the solved temperatures in addition to solver convergence. The provider/constitutive evidence is not silently replaced by the solver's numerical status.

## Compressor maps

`CompressorMapPlan` stores corrected-speed and operating-line axes plus corrected-flow, pressure-ratio, and isentropic-efficiency tables. It uses native rectilinear interpolation and returns explicit support. Off-map points are unsupported; constant-boundary interpolation is never treated as physical extrapolation.

`CompressorPlan.design` creates an immutable calibration artifact tied to its exact map. Compressor evaluation solves the isentropic and actual outlet states through the selected homogeneous thermodynamic model, conserves mass flow, and reports shaft power. Total gas stations remain distinct from generic process ports.
