# Vortex architecture

Phydrax separates vortex source semantics from induced-field execution.
`VortexSourceState` carries positions, integrated scalar circulation in 2-D or
integrated vector vorticity in 3-D, runtime activity, and optional core/volume.
These quantities never reuse `ParticleDiscretization.masses`.

`VortexTargetState` carries independent target coordinates and optional source
indices for identity-based self exclusion. Coincident distinct sources remain
real interactions; coordinate equality never implies self identity.

## Capability preparation

Every canonical velocity backend publishes `VortexVelocityCapabilities`:

- source kind and dimension;
- required source fields;
- supported velocity, gradient, and vorticity requests;
- free-space, periodic, bounded, or mixed domain;
- target topology;
- precision and derivative scope;
- acceleration identity.

`VortexVelocityCompatibility` binds this contract to exact source/target
capacities before compilation. Diffusion backends publish the analogous
`VortexDiffusionCapabilities`. `vortex_property_requirements` makes core and
volume requirements capability-driven rather than global.

## Continuous and event state

`VortexParticleMethodPlan` composes velocity, diffusion, and a formulation.
Classic VPM retains fixed core state. `ReformulatedVPMFormulation` places core
radius in the packed differential state and evolves it with strength.
Relaxation is an accepted-stage schedule, never hidden inside a field backend.

Topology changes use `VortexPopulationTransition`: candidate state, accepted
state, stable IDs, lineage, journal, strength/impulse evidence, and explicit
failure. Capacity growth starts a new compiled epoch; `VortexReplayPlan` and
`VortexTransitionPullback` join epochs without pretending event selection is
smooth. Transversal deterministic events may use `VortexSaltationMap`.

## Valid derivatives

Fixed source/target programs support ordinary JVP/VJP according to backend
capabilities. Remeshing, split/merge, reconnection, hierarchy rebuild, and
capacity growth expose transition pullbacks or an explicit
`UndefinedTopologyDerivative`. No straight-through topology estimator is used.
