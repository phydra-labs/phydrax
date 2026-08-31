# Causal circuit dynamics

`phydrax.circuit` lowers grounded nodal topology into the native Phydrax differential-algebraic substrate. A causal element law evaluates terminal currents and auxiliary residuals from time, terminal voltages and rates, internal state and rates, and explicit inputs. The resulting global state contains non-ground node voltages followed by ordered element state blocks.

`prepare_circuit_dae` compiles KCL routes, differential/algebraic roles, scales, and the exact residual into `phydrax.dynamics.DifferentialAlgebraicSystem`. Integration, consistent initialization, replay, adaptive stepping, and implicit differentiation remain owned by `phydrax.solver`.

Built-in resistor, capacitor, and inductor models have causal implicit lowerings. `CircuitElement` composes custom implicit, frequency, energy, and noise laws. Frequency-only components fail closed when DAE compilation is requested.

## Analysis consistency

The same implicit residual is used for:

- Zero-rate operating points.
- Transient DAE solves.
- Descriptor linearization.
- Harmonic balance.

No hidden GMIN, pseudoinverse, or source limiting is added. Source continuation is an explicit native continuation problem and its endpoint is evaluated against the physical residual.

## Evidence and boundaries

Circuit DAE diagnostics report the original residual, KCL residual, element residuals, terminal power, and finiteness. Floating topology, unsupported element laws, inconsistent state layouts, and DAE regularity failures are errors or native failure statuses. Topology changes, switch events, state-role changes, and singular branch transitions are nondifferentiable boundaries.
