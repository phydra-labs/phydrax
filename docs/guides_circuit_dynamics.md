# Causal circuit dynamics

`phydrax.circuit` lowers grounded nodal topology into the native Phydrax differential-algebraic substrate. A causal element law evaluates terminal currents and auxiliary residuals from time, terminal voltages and rates, internal state and rates, and explicit inputs. The resulting global state contains non-ground node voltages followed by ordered element state blocks.

`prepare_circuit_dae` compiles KCL routes, differential/algebraic roles, scales, and the exact residual into `phydrax.dynamics.DifferentialAlgebraicSystem`. Integration, consistent initialization, replay, adaptive stepping, and implicit differentiation remain owned by `phydrax.solver`.

Built-in resistor, capacitor, and inductor models have causal implicit lowerings and
explicit passive energy laws. `CircuitElement` composes custom implicit, frequency,
energy, and noise laws. Frequency-only components fail closed when DAE compilation is
requested.

## Analysis consistency

The same implicit residual is used for:

- Zero-rate operating points.
- Transient DAE solves.
- Descriptor linearization.
- Harmonic balance.

Independent-source input values may be scalars or explicit callables of DAE time; the
same callable is reused by transient diagnostics and harmonic balance.

No hidden GMIN, pseudoinverse, or source limiting is added. Source continuation is an explicit native continuation problem and its endpoint is evaluated against the physical residual.

## Energy evidence and boundaries

`evaluate_circuit_energy_ledger` evaluates a supplied time trajectory without replacing
the DAE. For passive elements only, it obtains stored energy and dissipated power from
`AbstractCircuitEnergyLaw`; built-in R, C, and L elements provide the corresponding
nonnegative dissipation or storage laws. Independent current and voltage sources are
never passed through the passive-energy interface. Their signed terminal power is
retained separately, where positive means absorbed by the source and negative means
supplied.

Port currents must be supplied explicitly to the audit because nodal ports are boundary
metadata, not transient DAE branches. They use the MNA convention (current into the
circuit), so the ledger stores the opposite signed power absorbed by the external port.
With device-terminal currents positive into each device, the checked equation is
`dW/dt + P_diss + P_ports + P_sources = 0`. Contributions, pointwise residual, endpoint
storage change, trapezoidal work integrals, interval defect, and nonnegative passive
dissipation evidence are all retained independently. `assess_circuit_energy_ledger`
recomputes closure from those retained terms.

An element without an explicit passive energy law, or a trajectory without required
port currents, yields unavailable evidence with named reasons rather than a fabricated
balance. `CircuitDAEDiagnostics.terminal_power` remains only the instantaneous algebraic
sum of device terminal products; it is not an energy-closure claim.

Floating topology, unsupported element laws, inconsistent state layouts, and DAE
regularity failures are errors or native failure statuses. Topology changes, switch
events, state-role changes, and singular branch transitions are nondifferentiable
boundaries.
