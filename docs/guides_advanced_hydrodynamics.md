# Advanced free-surface hydrodynamics

This guide extends the fixed-topology one-phase graph ALE product with variational
surface tension, coherent wave forcing and absorption, conservative vertical rezoning,
shoreline event handling, and submerged rigid/hydroelastic bodies.

## Corrected baseline contracts

The advanced product uses one pressure reference:

`Pi = p_liquid / rho + g z - p_reference / rho`.

With liquid-to-gas normal and `p_liquid - p_gas = sigma kappa`, the top head is

`Pi_surface = g eta + (p_gas - p_reference) / rho + h_capillary + h_wave`.

Boundary layout and stage values are owned by `FreeSurfaceBoundaryPlan`; projection no
longer creates an independent mask. The same mapped kinetic functional defines state
momentum, inverse Hodge, projection, diagnostics, remap evidence, and body work.

The public midpoint method evaluates midpoint physics at `t + dt/2` and endpoint
geometry at `t + dt`. Pressure work is integrated from stage pressure head and top
volume flux rather than inferred from endpoint energy.

## Variational graph surface tension

`GraphCapillarityPlan` defines surface energy from one fixed triangulation of the mapped
top surface. The generalized capillary force is the derivative of that discrete area.
Capillary pressure head is obtained through the transpose of the exact surface-volume
Jacobian, so capillary work is dual to the kinematic eta-rate map.

The result reports surface area/energy, generalized force, pressure head, dual residual,
and explicit capillary timestep limit. Contact lines are not part of the graph-capillary
route.

## Incident waves, generation, and absorption

`IncidentWavePlan` owns coherent regular or irregular linear components, explicit
phases, current convention, ramp, dispersion roots, velocity, pressure head, and energy
flux.

`WaveForcingPlan` separates:

- fixed-boundary prescribed velocity;
- compatible top pressure target;
- surface target diagnostics while eta-rate remains top-flux constrained;
- volumetric relaxation forcing;
- sponge dissipation;
- active absorption history/controller.

A closed tank may remove mean normal return flow. Controller and irregular-wave phase
history are restart state.

## Vertical rezoning and shoreline events

`FreeSurfaceRezonePlan` changes only the interior vertical reference distribution. It
keeps horizontal footprint, connectivity, eta, and bottom fixed. Scalar content is
transferred by physical vertical overlap; physical velocity is interpolated and mapped
back through the new Hodge. Rezone evidence reports scalar, momentum, energy, quality,
and event identities.

`GraphShorelineEventPlan` does not create dry columns in graph ALE. It emits explicit
continue/rezone/handoff/reject status. Small depth, excessive graph slope, topology
change, or multivalued interface routes to hydrostatic wet/dry or two-phase VOF rather
than clipping graph height.

## Rigid and hydroelastic bodies

`MappedRigidHydroelasticBodyPlan` supplies mapped marker gather/spread by exact linear
transpose, fully submerged normal constraints, rigid mass/inertia response, modal
response, drag evidence, and action/reaction work.

`RigidHydroelasticALEMethod` couples the free-surface step to the marker/body Schur
response and records body work and structural energy. Its graph epoch remains fully
submerged with fixed marker topology. `GraphShorelineEventPlan` hands surface piercing,
contact, breaking, overturning, or multivalued geometry to the canonical two-phase VOF
product rather than extending graph eta past its mathematical domain.

## Canonical topology-changing product

`IncompressibleTwoPhaseVOFPlan` owns variable-density projection, two- and
three-dimensional PLIC/CLSVOF, contact angle, surface tension, wetting/drying, moving
bodies, and topology-changing interfaces. `TwoPhaseCapabilityEventPlan` names the
surface-piercing, moving-contact, body-contact, wet/dry, breaking, and overturning
routes with explicit derivative boundaries. `ConservativeTwoPhaseRemeshPlan` transfers
extensive phase/scalar content and face momentum between preflighted mesh epochs. This
is one canonical product boundary: the graph ALE solver is not duplicated into a
second interface-capturing implementation.

## Work ledger and restart

The expanded graph ledger names:

- kinetic, gravity, and surface-energy changes;
- noncapillary pressure work;
- capillary work;
- gas-pressure work;
- wave/relaxation/controller work;
- sponge dissipation;
- body work/energy;
- remap, shoreline, GCL, divergence, kinematic, dynamic, capillary-dual, and nonlinear
  residuals.

Checkpoints use the corrected pressure/boundary/work/capillary/mesh-epoch schema and
fingerprint the complete bottom array.

## Limits

- graph topology remains fixed within a graph ALE epoch;
- graph rezoning changes only the vertical reference distribution;
- general wetting/drying, surface piercing, contact, breaking, and overturning use an
  explicit handoff to the two-phase product;
- moving wavemakers require lateral ALE map support;
- event routing, remeshing, and topology changes are not ordinarily differentiable.
